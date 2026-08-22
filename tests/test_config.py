"""Tests for agent.config — Settings loading and validation."""

from agent.config import Settings


class TestSettingsDefaults:
    """Tests for Settings default values."""

    def test_default_provider(self):
        s = Settings()
        assert s.LLM_PROVIDER == "openai"

    def test_default_temperatures(self):
        s = Settings()
        assert isinstance(s.OPENAI_TEMPERATURE, float)
        assert s.OPENAI_TEMPERATURE == 0.1

    def test_default_int_fields(self):
        s = Settings()
        assert s.MAX_ITERATIONS > 0
        assert s.MAX_EXECUTION_TIME_SEC > 0
        assert s.MAX_SHORT_TERM_ROUNDS > 0
        assert s.MAX_RETRIES >= 0

    def test_bool_field_is_bool(self):
        s = Settings()
        assert isinstance(s.BROWSER_HEADLESS, bool)

    def test_get_llm_config(self):
        s = Settings()
        config = s.get_llm_config()
        assert config["provider"] == "openai"
        assert "model" in config
        assert "temperature" in config

    def test_get_llm_config_returns_dict(self):
        s = Settings()
        config = s.get_llm_config()
        assert "provider" in config
        assert "model" in config
        assert "temperature" in config

    def test_empty_api_key(self):
        s = Settings()
        s.OPENAI_API_KEY = ""
        assert s.OPENAI_API_KEY == ""

    def test_file_tool_defaults(self):
        s = Settings()
        assert s.FILE_READER_ROOT == "."
        assert s.NOTES_DIR == "./agent_notes"
        assert s.CODE_WORKDIR == "./agent_workspace"


class TestGlobalSettings:
    """Tests for the global settings singleton."""

    def test_global_settings_exists(self):
        from agent.config import settings
        assert isinstance(settings, Settings)

    def test_global_settings_has_provider(self):
        from agent.config import settings
        assert settings.LLM_PROVIDER == "openai"