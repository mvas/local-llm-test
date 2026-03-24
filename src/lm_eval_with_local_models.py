from lm_eval.__main__ import cli_evaluate

# Import for registration side effects before CLI model lookup.
import lm_eval_llamacpp_native_mc  # noqa: F401


if __name__ == "__main__":
    cli_evaluate()
