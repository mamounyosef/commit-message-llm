#!/usr/bin/env python3
"""
Inference script for commit message LLM.

This script generates commit messages from Git diffs using a trained model.
"""

import json
import sys
from pathlib import Path

# Set up default config path
config_path = Path(__file__).parent / "config.yaml"
if config_path.exists():
    sys.path.insert(0, str(Path(__file__).parent))
    from commit_message_llm.config import set_default_config_path
    set_default_config_path(config_path)

from commit_message_llm.cli import parse_args_infer
from commit_message_llm.config import load_config, ConfigSchema
from commit_message_llm.inference import CommitMessageGenerator
from commit_message_llm.utils import setup_logging
from commit_message_llm.utils.logging_config import get_logger


logger = get_logger(__name__)


def main() -> int:
    """
    Main inference function.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    # Parse arguments
    args = parse_args_infer()

    # Load config if available
    config = None
    try:
        config = load_config()
    except FileNotFoundError:
        logger.debug("No config file found, using defaults")

    # Set up logging
    log_level = "WARNING" if not args.verbose else "INFO"
    setup_logging(level=log_level, log_to_console=True)

    # Initialize generator
    device = None if args.device == "auto" else args.device
    generator = CommitMessageGenerator(
        model_path=args.model,
        config=config,
        device=device,
        load_adapter=not args.no_adapter,
    )

    # Load model
    generator.load()

    # Get input diff
    diff_text = None

    if args.diff:
        diff_text = args.diff
    elif args.diff_file:
        with open(args.diff_file, "r", encoding="utf-8") as f:
            diff_text = f.read()
    elif args.git:
        import subprocess
        result = subprocess.run(
            ["git", "diff"],
            capture_output=True,
            text=True,
            check=False,
        )
        diff_text = result.stdout
        if not diff_text:
            logger.error("No git diff found")
            return 1
    elif args.interactive:
        logger.info("Enter diff text (Ctrl+D to finish):")
        diff_text = sys.stdin.read()

    if not diff_text:
        logger.error("No diff input provided")
        return 1

    # Generate commit message
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
    }

    if args.do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    result = generator.generate(diff_text, **generation_kwargs)

    # Output result
    if args.json:
        output = {
            "commit_message": result.commit_message,
            "generation_time": result.generation_time,
        }
        if args.verbose:
            output["diff"] = result.diff
        print(json.dumps(output, indent=2))
    else:
        if args.verbose:
            print("=" * 80)
            print("Diff:")
            print(result.diff[:800] + ("..." if len(result.diff) > 800 else ""))
            print()
        print("Commit Message:")
        print(result.commit_message)
        if args.verbose:
            print()
            print(f"Generated in {result.generation_time:.2f} seconds")

    # Save to file if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            if args.json:
                json.dump({
                    "commit_message": result.commit_message,
                    "generation_time": result.generation_time,
                }, f, indent=2)
            else:
                f.write(result.commit_message)
        logger.info(f"Output saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
