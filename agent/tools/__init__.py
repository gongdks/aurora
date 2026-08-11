"""Tool ecosystem — import each module to trigger @register."""

from agent.tools import calculator    # noqa: F401
from agent.tools import web_search    # noqa: F401
from agent.tools import file_reader   # noqa: F401
from agent.tools import file_writer   # noqa: F401
from agent.tools import file_opener   # noqa: F401
from agent.tools import code_executor # noqa: F401
from agent.tools import datetime_tool # noqa: F401
from agent.tools import web_fetcher   # noqa: F401
from agent.tools import web_content_fetcher  # noqa: F401
from agent.tools import note_manager  # noqa: F401
from agent.tools import code_search   # noqa: F401
from agent.tools import code_analysis # noqa: F401
from agent.tools import browser       # noqa: F401