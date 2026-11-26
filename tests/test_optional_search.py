import subprocess
import sys
from textwrap import dedent


def test_core_and_search_imports_work_without_sentence_transformers():
    subprocess.run(
        [
            sys.executable,
            "-c",
            dedent("""\
                import sys

                sys.modules["sentence_transformers"] = None

                from gleefultransformer import CHATCallArgs, GleefulChat, GleefulRanker, GleefulSearch
                from gleefultransformer.gleefulsearch import SearchCallArgs

                assert isinstance(CHATCallArgs().search_args, SearchCallArgs)

                for create in (
                    GleefulRanker,
                    GleefulSearch,
                    lambda: GleefulSearch(gleeful_ranker=object()),
                ):
                    try:
                        create()
                    except ImportError as error:
                        assert "pip install 'gleefultransformer[search]'" in str(error)
                    else:
                        raise AssertionError("Expected the search extra to be required")
            """),
        ],
        check=True,
    )
