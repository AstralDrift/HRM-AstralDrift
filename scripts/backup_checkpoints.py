#!/usr/bin/env python3
"""
Comprehensive checkpoint backup system before restart
"""

import os
import shutil
import json
import hashlib
from pathlib import Path
from datetime import datetime
import tarfile
import argparse

class CheckpointBackupManager:
    def __init__(self, source_dir="checkpoints", backup_root="backups"):
        self.source_dir = Path(source_dir)
        self.backup_root = Path(backup_root)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create backup root directory
        self.backup_root.mkdir(exist_ok=True)
        
    def calculate_file_hash(self, file_path):
        """Calculate SHA256 hash of file for integrity verification"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def scan_checkpoints(self):
        """Scan for existing checkpoints and analyze them"""
        if not self.source_dir.exists():
            return {"status": "no_source", "message": f"Source directory {self.source_dir} not found"}
        
        checkpoints = []
        total_size = 0
        
        # Find all checkpoint files
        for pattern in ["*.pt", "*.pth", "*.ckpt"]:
            for file_path in self.source_dir.rglob(pattern):
                if file_path.is_file():
                    file_stats = file_path.stat()
                    file_size = file_stats.st_size
                    total_size += file_size
                    
                    # Try to extract epoch info from filename or file content
                    epoch_info = self.extract_epoch_info(file_path)
                    
                    checkpoints.append({
                        "path": str(file_path),
                        "relative_path": str(file_path.relative_to(self.source_dir)),
                        "size_mb": file_size / (1024 * 1024),
                        "modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                        "epoch": epoch_info.get("epoch"),
                        "hash": self.calculate_file_hash(file_path)
                    })
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: x["modified"], reverse=True)
        
        return {
            "status": "success",
            "checkpoint_count": len(checkpoints),
            "total_size_mb": total_size / (1024 * 1024),
            "checkpoints": checkpoints
        }
    
    def extract_epoch_info(self, checkpoint_path):
        """Extract epoch information from checkpoint"""
        try:
            # Try to load checkpoint and extract epoch
            import torch
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            epoch = checkpoint.get('epoch', None)
            loss = checkpoint.get('loss', None)
            
            return {"epoch": epoch, "loss": loss}
        except Exception:
            # Fallback: try to extract from filename
            filename = checkpoint_path.name
            if 'epoch' in filename.lower():
                import re
                match = re.search(r'epoch[_-]?(\d+)', filename.lower())
                if match:
                    return {"epoch": int(match.group(1))}
            return {"epoch": None}
    
    def create_incremental_backup(self, backup_name=None):
        """Create incremental backup of checkpoints"""
        if backup_name is None:
            backup_name = f"checkpoint_backup_{self.timestamp}"
        
        backup_dir = self.backup_root / backup_name
        backup_dir.mkdir(exist_ok=True)
        
        print(f"📦 Creating incremental backup: {backup_dir}")
        
        # Scan current checkpoints
        scan_result = self.scan_checkpoints()
        if scan_result["status"] != "success":
            return scan_result
        
        checkpoints = scan_result["checkpoints"]
        
        # Load previous backup manifest if exists
        previous_manifest = self.load_previous_manifest()
        
        # Determine which files need backup
        files_to_backup = []
        skipped_files = []
        
        for checkpoint in checkpoints:
            file_path = Path(checkpoint["path"])
            file_hash = checkpoint["hash"]
            
            # Check if file changed since last backup
            needs_backup = True
            if previous_manifest:
                prev_file = next((f for f in previous_manifest.get("files", []) 
                                if f["relative_path"] == checkpoint["relative_path"]), None)
                if prev_file and prev_file["hash"] == file_hash:
                    needs_backup = False
                    skipped_files.append(checkpoint)
            
            if needs_backup:
                files_to_backup.append(checkpoint)
        
        print(f"📊 Backup Analysis:")
        print(f"   Total checkpoints: {len(checkpoints)}")
        print(f"   Files to backup: {len(files_to_backup)}")
        print(f"   Unchanged files: {len(skipped_files)}")
        
        # Copy files that need backup
        backup_manifest = {
            "backup_name": backup_name,
            "timestamp": self.timestamp,
            "source_dir": str(self.source_dir),
            "backup_type": "incremental",
            "files": [],
            "stats": {
                "total_files": len(checkpoints),
                "backed_up_files": len(files_to_backup),
                "skipped_files": len(skipped_files),
                "total_size_mb": 0
            }
        }
        
        total_copied_size = 0
        
        for checkpoint in files_to_backup:
            source_path = Path(checkpoint["path"])
            target_path = backup_dir / checkpoint["relative_path"]
            
            # Create target directory if needed
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            print(f"   Copying: {checkpoint['relative_path']} ({checkpoint['size_mb']:.1f} MB)")
            shutil.copy2(source_path, target_path)
            
            # Verify copy
            copied_hash = self.calculate_file_hash(target_path)
            if copied_hash != checkpoint["hash"]:
                raise RuntimeError(f"Hash mismatch for {checkpoint['relative_path']}")
            
            total_copied_size += checkpoint["size_mb"]
            backup_manifest["files"].append(checkpoint)
        
        backup_manifest["stats"]["total_size_mb"] = total_copied_size
        
        # Save manifest
        manifest_path = backup_dir / "backup_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(backup_manifest, f, indent=2)
        
        # Update latest backup reference
        latest_backup_path = self.backup_root / "latest_backup.json"
        with open(latest_backup_path, 'w') as f:
            json.dump({
                "backup_name": backup_name,
                "backup_dir": str(backup_dir),
                "timestamp": self.timestamp,
                "manifest": backup_manifest
            }, f, indent=2)
        
        print(f"✅ Backup completed: {backup_dir}")
        print(f"💾 Copied {len(files_to_backup)} files ({total_copied_size:.1f} MB)")
        
        return {
            "status": "success",
            "backup_dir": str(backup_dir),
            "manifest": backup_manifest
        }
    
    def create_compressed_backup(self, backup_name=None):
        """Create compressed tar.gz backup"""
        if backup_name is None:
            backup_name = f"checkpoint_archive_{self.timestamp}"
        
        archive_path = self.backup_root / f"{backup_name}.tar.gz"
        
        print(f"🗜️ Creating compressed backup: {archive_path}")
        
        # Scan checkpoints
        scan_result = self.scan_checkpoints()
        if scan_result["status"] != "success":
            return scan_result
        
        # Create tar.gz archive
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(self.source_dir, arcname="checkpoints")
        
        # Calculate archive size
        archive_size = archive_path.stat().st_size / (1024 * 1024)
        original_size = scan_result["total_size_mb"]
        compression_ratio = (1 - archive_size / original_size) * 100 if original_size > 0 else 0
        
        # Create manifest
        manifest = {
            "backup_name": backup_name,
            "timestamp": self.timestamp,
            "backup_type": "compressed",
            "archive_path": str(archive_path),
            "source_dir": str(self.source_dir),
            "stats": {
                "original_size_mb": original_size,
                "archive_size_mb": archive_size,
                "compression_ratio": compression_ratio,
                "checkpoint_count": scan_result["checkpoint_count"]
            },
            "checkpoints": scan_result["checkpoints"]
        }
        
        # Save manifest
        manifest_path = self.backup_root / f"{backup_name}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"✅ Compressed backup completed")
        print(f"📦 Archive size: {archive_size:.1f} MB")
        print(f"📊 Compression: {compression_ratio:.1f}% smaller")
        
        return {
            "status": "success",
            "archive_path": str(archive_path),
            "manifest": manifest
        }
    
    def load_previous_manifest(self):
        """Load previous backup manifest for incremental backup"""
        latest_backup_path = self.backup_root / "latest_backup.json"
        if not latest_backup_path.exists():
            return None
        
        try:
            with open(latest_backup_path, 'r') as f:
                return json.load(f).get("manifest")
        except:
            return None
    
    def list_backups(self):
        """List all existing backups"""
        if not self.backup_root.exists():
            return {"status": "no_backups", "backups": []}
        
        backups = []
        
        # Find backup directories
        for item in self.backup_root.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                manifest_path = item / "backup_manifest.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path, 'r') as f:
                            manifest = json.load(f)
                        backups.append({
                            "type": "directory",
                            "name": item.name,
                            "path": str(item),
                            "manifest": manifest
                        })
                    except:
                        pass
        
        # Find compressed backups
        for item in self.backup_root.glob("*.tar.gz"):
            manifest_path = self.backup_root / f"{item.stem}_manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    backups.append({
                        "type": "archive",
                        "name": item.name,
                        "path": str(item),
                        "manifest": manifest
                    })
                except:
                    pass
        
        # Sort by timestamp
        backups.sort(key=lambda x: x["manifest"].get("timestamp", ""), reverse=True)
        
        return {
            "status": "success",
            "backup_count": len(backups),
            "backups": backups
        }
    
    def restore_backup(self, backup_name, target_dir=None):
        """Restore from backup (placeholder - implement as needed)"""
        print(f"🔄 Restore functionality for {backup_name}")
        print("⚠️ Restore not implemented - manual restoration recommended for safety")
        return {"status": "not_implemented"}

def main():
    """Main backup function"""
    parser = argparse.ArgumentParser(description="HRM Checkpoint Backup Manager")
    parser.add_argument("--source-dir", default="checkpoints", help="Source checkpoint directory")
    parser.add_argument("--backup-root", default="backups", help="Backup root directory")
    parser.add_argument("--action", choices=["scan", "backup", "compress", "list"], 
                       default="backup", help="Action to perform")
    parser.add_argument("--name", help="Backup name (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    manager = CheckpointBackupManager(args.source_dir, args.backup_root)
    
    if args.action == "scan":
        result = manager.scan_checkpoints()
        print(json.dumps(result, indent=2, default=str))
    
    elif args.action == "backup":
        result = manager.create_incremental_backup(args.name)
        if result["status"] == "success":
            print("🎉 Incremental backup completed successfully")
        else:
            print(f"❌ Backup failed: {result}")
    
    elif args.action == "compress":
        result = manager.create_compressed_backup(args.name)
        if result["status"] == "success":
            print("🎉 Compressed backup completed successfully")
        else:
            print(f"❌ Compressed backup failed: {result}")
    
    elif args.action == "list":
        result = manager.list_backups()
        print(f"\n📋 Found {result['backup_count']} backups:")
        for backup in result["backups"]:
            manifest = backup["manifest"]
            print(f"   {backup['type'].upper()}: {backup['name']}")
            print(f"      Created: {manifest.get('timestamp', 'Unknown')}")
            if 'stats' in manifest:
                stats = manifest['stats']
                size_key = 'archive_size_mb' if backup['type'] == 'archive' else 'total_size_mb'
                size = stats.get(size_key, stats.get('total_size_mb', 0))
                print(f"      Size: {size:.1f} MB")
                print(f"      Files: {stats.get('checkpoint_count', stats.get('total_files', 0))}")
            print()

if __name__ == "__main__":
    main()