# Copyright (C) 2026 Teodor Smith
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import json
import logging
from pathlib import Path
from rich.console import Console
from rich.table import Table

from agent.engine.trainer import ModelTrainer
from agent.config import DATA_DIR

logger = logging.getLogger(__name__)
console = Console()

def handle_train_cmd(rest: str, trainer: ModelTrainer) -> None:
    parts = rest.split()
    if not parts:
        console.print("[yellow]Usage: train <dpo|list> [options][/yellow]")
        return
        
    subcmd = parts[0].lower()
    
    if subcmd == "list":
        checkpoint_dir = DATA_DIR / "checkpoints"
        if not checkpoint_dir.exists():
            console.print("No checkpoints found.")
            return
            
        table = Table(title="Available LoRA Checkpoints")
        table.add_column("Version", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Dataset", style="green")
        
        has_rows = False
        for d in sorted(checkpoint_dir.iterdir(), key=lambda x: x.name):
            if d.is_dir() and d.name.startswith("lora_v"):
                has_rows = True
                meta_file = d / "adapter_meta.json"
                status = "unknown"
                dataset = "unknown"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r") as f:
                            meta = json.load(f)
                            status = meta.get("status", status)
                            dataset = Path(meta.get("dataset", dataset)).name
                    except Exception:
                        pass
                table.add_row(d.name, status, dataset)
                
        if has_rows:
            console.print(table)
        else:
            console.print("No checkpoints found.")
        
    elif subcmd == "dpo":
        dataset_path = None
        epochs = 1
        batch_size = 2
        dry_run = False
        model_id = None
        
        args = parts[1:]
        try:
            if "--dataset" in args:
                idx = args.index("--dataset")
                dataset_path = args[idx+1]
            if "--epochs" in args:
                idx = args.index("--epochs")
                epochs = int(args[idx+1])
            if "--batch-size" in args:
                idx = args.index("--batch-size")
                batch_size = int(args[idx+1])
            if "--model-id" in args:
                idx = args.index("--model-id")
                model_id = args[idx+1]
            if "--dry-run" in args:
                dry_run = True
        except (ValueError, IndexError):
            console.print("[red]Invalid arguments for train dpo.[/red]")
            return
            
        if not dry_run:
            console.print("[bold yellow]WARNING: This will download a base model and consume GPU resources.[/bold yellow]")
            resp = input("Proceed? [y/N]: ").strip().lower()
            if resp != "y":
                console.print("Training aborted.")
                return
                
        console.print("[bold cyan]Starting QLoRA DPO Fine-tuning...[/bold cyan]")
        try:
            out_dir = trainer.train_dpo(
                dataset_path=dataset_path, 
                epochs=epochs, 
                batch_size=batch_size,
                dry_run=dry_run,
                model_id=model_id
            )
            console.print(f"[green]Training completed. Artifacts saved to: {out_dir}[/green]")
        except Exception as e:
            console.print(f"[bold red]Training failed:[/bold red] {e}")
            logger.exception("Training failed")
            
    elif subcmd == "promote":
        if len(parts) < 2:
            console.print("[red]Usage: train promote <lora_vX>[/red]")
            return
        version = parts[1]
        meta_file = DATA_DIR / "checkpoints" / version / "adapter_meta.json"
        if not meta_file.exists():
            console.print(f"[red]Checkpoint {version} not found.[/red]")
            return
        
        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)
            meta["status"] = "promoted"
            with open(meta_file, "w") as f:
                json.dump(meta, f, indent=2)
            console.print(f"[green]Successfully promoted {version}. MoA router will now use this adapter.[/green]")
        except Exception as e:
            console.print(f"[red]Failed to promote: {e}[/red]")
            
    elif subcmd == "evaluate":
        if len(parts) < 2:
            console.print("[red]Usage: train evaluate <lora_vX>[/red]")
            return
        version = parts[1]
        console.print(f"[cyan]Evaluating {version}...[/cyan]")
        console.print("[yellow]Evaluation benchmark suite is currently a manual stub. Please verify the model qualitatively against the immutable suite before promotion.[/yellow]")
        
    elif subcmd == "export-ollama":
        if len(parts) < 2:
            console.print("[red]Usage: train export-ollama <lora_vX>[/red]")
            return
        version = parts[1]
        checkpoint_dir = DATA_DIR / "checkpoints" / version
        if not checkpoint_dir.exists():
            console.print(f"[red]Checkpoint {version} not found.[/red]")
            return
            
        console.print(f"[cyan]Merging {version} into base model for Ollama export (this may take a few minutes)...[/cyan]")
        try:
            meta_file = checkpoint_dir / "adapter_meta.json"
            base_model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
            if meta_file.exists():
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                    base_model_id = meta.get("base_model", base_model_id)
                    
            from transformers import AutoModelForCausalLM
            from peft import PeftModel
            import torch
            
            console.print(f"Loading base model: {base_model_id}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id, torch_dtype=torch.float16, device_map="cpu"
            )
            console.print("Loading PEFT adapter and merging...")
            model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
            merged_model = model.merge_and_unload()
            
            out_dir = checkpoint_dir / "merged"
            out_dir.mkdir(exist_ok=True)
            merged_model.save_pretrained(str(out_dir))
            
            modelfile_path = checkpoint_dir / "Modelfile"
            with open(modelfile_path, "w") as f:
                f.write(f"FROM ./merged\n")
                
            console.print("[green]Merge complete![/green]")
            console.print(f"To create the Ollama model, run:\n[bold cyan]ollama create {base_model_id.split('/')[-1].lower()}-dpo -f {modelfile_path}[/bold cyan]")
        except Exception as e:
            console.print(f"[bold red]Export failed:[/bold red] {e}")
            logger.exception("Ollama export failed")
            
    else:
        console.print(f"[red]Unknown train sub-command: {subcmd}[/red]")
