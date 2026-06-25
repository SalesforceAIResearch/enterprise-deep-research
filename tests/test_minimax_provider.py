"""Unit tests for MiniMax LLM provider integration."""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock langchain_google_vertexai before importing llm_clients
sys.modules["langchain_google_vertexai"] = MagicMock()


class TestMiniMaxModelConfigs:
    """Test MiniMax entries in MODEL_CONFIGS."""

    def test_minimax_in_model_configs(self):
        """MiniMax should be registered in MODEL_CONFIGS."""
        from llm_clients import MODEL_CONFIGS

        assert "minimax" in MODEL_CONFIGS

    def test_minimax_default_model(self):
        """MiniMax default model should be MiniMax-M2.7."""
        from llm_clients import MODEL_CONFIGS

        assert MODEL_CONFIGS["minimax"]["default_model"] == "MiniMax-M2.7"

    def test_minimax_available_models(self):
        """MiniMax should have M2.7 and M2.7-highspeed models."""
        from llm_clients import MODEL_CONFIGS

        models = MODEL_CONFIGS["minimax"]["available_models"]
        assert "MiniMax-M2.7" in models
        assert "MiniMax-M2.7-highspeed" in models

    def test_minimax_requires_api_key_field(self):
        """MiniMax config should have requires_api_key field."""
        from llm_clients import MODEL_CONFIGS

        assert "requires_api_key" in MODEL_CONFIGS["minimax"]


class TestMiniMaxClient:
    """Test MiniMaxClient class."""

    def test_client_initialization(self):
        """MiniMaxClient should initialize with correct attributes."""
        from llm_clients import MiniMaxClient

        client = MiniMaxClient(
            model_name="MiniMax-M2.7",
            api_key="test-key",
            max_tokens=16384,
        )
        assert client.model == "MiniMax-M2.7"
        assert client.model_name == "MiniMax-M2.7"
        assert client._max_tokens == 16384

    def test_temperature_clamping_zero(self):
        """Temperature 0.0 should be clamped to 0.01."""
        from llm_clients import MiniMaxClient

        assert MiniMaxClient._clamp_temperature(0.0) == 0.01

    def test_temperature_clamping_negative(self):
        """Negative temperature should be clamped to 0.01."""
        from llm_clients import MiniMaxClient

        assert MiniMaxClient._clamp_temperature(-0.5) == 0.01

    def test_temperature_clamping_high(self):
        """Temperature > 1.0 should be clamped to 1.0."""
        from llm_clients import MiniMaxClient

        assert MiniMaxClient._clamp_temperature(1.5) == 1.0

    def test_temperature_clamping_valid(self):
        """Valid temperature should pass through unchanged."""
        from llm_clients import MiniMaxClient

        assert MiniMaxClient._clamp_temperature(0.5) == 0.5
        assert MiniMaxClient._clamp_temperature(0.7) == 0.7
        assert MiniMaxClient._clamp_temperature(1.0) == 1.0

    def test_strip_think_tags(self):
        """Think tags should be stripped from responses."""
        from llm_clients import MiniMaxClient

        text = "<think>internal reasoning here</think>Final answer."
        assert MiniMaxClient._strip_think_tags(text) == "Final answer."

    def test_strip_think_tags_multiline(self):
        """Multiline think tags should be stripped."""
        from llm_clients import MiniMaxClient

        text = "<think>\nstep 1\nstep 2\n</think>\nThe result is 42."
        result = MiniMaxClient._strip_think_tags(text)
        assert "<think>" not in result
        assert "The result is 42." in result

    def test_strip_think_tags_no_tags(self):
        """Text without think tags should be unchanged."""
        from llm_clients import MiniMaxClient

        text = "Just a normal response."
        assert MiniMaxClient._strip_think_tags(text) == text

    def test_client_openai_base_url(self):
        """Client should use MiniMax's OpenAI-compatible base URL."""
        from llm_clients import MiniMaxClient

        client = MiniMaxClient(
            model_name="MiniMax-M2.7",
            api_key="test-key",
        )
        assert client._client.base_url.host == "api.minimax.io"


class TestMiniMaxInLLMProviderEnum:
    """Test MiniMax in LLMProvider enum."""

    def test_minimax_in_llm_provider_enum(self):
        """MINIMAX should be in LLMProvider enum."""
        from src.configuration import LLMProvider

        assert hasattr(LLMProvider, "MINIMAX")
        assert LLMProvider.MINIMAX.value == "minimax"


class TestGetLlmClientMiniMax:
    """Test get_llm_client() with minimax provider."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    def test_get_llm_client_minimax_default(self):
        """get_llm_client('minimax') should return MiniMaxClient with default model."""
        # Need to reload to pick up env var
        import importlib
        import llm_clients

        importlib.reload(llm_clients)
        from llm_clients import get_llm_client, MiniMaxClient

        client = get_llm_client("minimax")
        assert isinstance(client, MiniMaxClient)
        assert client.model_name == "MiniMax-M2.7"

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    def test_get_llm_client_minimax_custom_model(self):
        """get_llm_client('minimax', 'MiniMax-M2.7-highspeed') should use specified model."""
        import importlib
        import llm_clients

        importlib.reload(llm_clients)
        from llm_clients import get_llm_client, MiniMaxClient

        client = get_llm_client("minimax", "MiniMax-M2.7-highspeed")
        assert isinstance(client, MiniMaxClient)
        assert client.model_name == "MiniMax-M2.7-highspeed"

    @patch.dict(os.environ, {}, clear=False)
    def test_get_llm_client_minimax_no_key_raises(self):
        """get_llm_client('minimax') without API key should raise ValueError."""
        # Remove MINIMAX_API_KEY if set
        env = os.environ.copy()
        env.pop("MINIMAX_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            import importlib
            import llm_clients

            importlib.reload(llm_clients)
            from llm_clients import get_llm_client

            with pytest.raises(ValueError, match="MINIMAX_API_KEY"):
                get_llm_client("minimax")


class TestGetAvailableProvidersMiniMax:
    """Test get_available_providers() includes minimax when key is set."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    def test_minimax_in_available_providers(self):
        """MiniMax should appear in available providers when API key is set."""
        import importlib
        import llm_clients

        importlib.reload(llm_clients)
        from llm_clients import get_available_providers

        providers = get_available_providers()
        assert "minimax" in providers


class TestConfigurationMiniMax:
    """Test Configuration class with MiniMax provider."""

    @patch.dict(os.environ, {"LLM_PROVIDER": "minimax"})
    def test_configuration_minimax_provider(self):
        """Configuration should accept minimax as LLM provider."""
        from src.configuration import Configuration, LLMProvider

        config = Configuration()
        assert config.llm_provider == LLMProvider.MINIMAX

    @patch.dict(os.environ, {"LLM_PROVIDER": "minimax"})
    def test_configuration_minimax_default_model(self):
        """Configuration should return MiniMax-M2.7 as default model for minimax."""
        from src.configuration import Configuration

        config = Configuration()
        assert config.llm_model == "MiniMax-M2.7"

    def test_configuration_minimax_explicit(self):
        """Configuration should accept minimax via kwargs."""
        from src.configuration import Configuration, LLMProvider

        config = Configuration(llm_provider=LLMProvider.MINIMAX, llm_model="MiniMax-M2.7-highspeed")
        assert config.llm_provider == LLMProvider.MINIMAX
        assert config.llm_model == "MiniMax-M2.7-highspeed"


class TestMiniMaxTokenLimits:
    """Test MiniMax token limit constants."""

    def test_minimax_max_tokens_defined(self):
        """MINIMAX_MAX_TOKENS should be defined."""
        from llm_clients import MINIMAX_MAX_TOKENS

        assert MINIMAX_MAX_TOKENS == 16384

    def test_minimax_max_tokens_positive(self):
        """MINIMAX_MAX_TOKENS should be a positive integer."""
        from llm_clients import MINIMAX_MAX_TOKENS

        assert isinstance(MINIMAX_MAX_TOKENS, int)
        assert MINIMAX_MAX_TOKENS > 0


class TestMiniMaxClientInvoke:
    """Test MiniMaxClient.invoke() with mocked API."""

    @patch("openai.OpenAI")
    def test_invoke_with_langchain_messages(self, mock_openai_cls):
        """invoke() should convert LangChain messages and return response text."""
        from llm_clients import MiniMaxClient

        # Setup mock
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Mock streaming response
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta = MagicMock()
        mock_chunk.choices[0].delta.content = "Hello from MiniMax"

        mock_client.chat.completions.create.return_value = [mock_chunk]

        client = MiniMaxClient(model_name="MiniMax-M2.7", api_key="test-key")
        client._client = mock_client

        # Create mock LangChain messages
        system_msg = MagicMock()
        system_msg.type = "system"
        system_msg.content = "You are a helpful assistant."
        human_msg = MagicMock()
        human_msg.type = "human"
        human_msg.content = "Hello!"

        result = client.invoke([system_msg, human_msg])

        assert result == "Hello from MiniMax"
        mock_client.chat.completions.create.assert_called_once()

    @patch("openai.OpenAI")
    def test_invoke_strips_think_tags(self, mock_openai_cls):
        """invoke() should strip <think> tags from response."""
        from llm_clients import MiniMaxClient

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Mock response with think tags
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta = MagicMock()
        mock_chunk1.choices[0].delta.content = "<think>reasoning</think>"

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta = MagicMock()
        mock_chunk2.choices[0].delta.content = "Final answer."

        mock_client.chat.completions.create.return_value = [mock_chunk1, mock_chunk2]

        client = MiniMaxClient(model_name="MiniMax-M2.7", api_key="test-key")
        client._client = mock_client

        msg = MagicMock()
        msg.type = "human"
        msg.content = "Test"

        result = client.invoke([msg])

        assert "<think>" not in result
        assert "Final answer." in result


class TestMiniMaxIntegration:
    """Integration tests for MiniMax provider (require MINIMAX_API_KEY)."""

    @pytest.fixture(autouse=True)
    def skip_without_api_key(self):
        if not os.getenv("MINIMAX_API_KEY"):
            pytest.skip("MINIMAX_API_KEY not set, skipping integration tests")

    def test_minimax_live_invoke(self):
        """Integration: invoke MiniMax API and get a response."""
        import importlib
        import llm_clients

        importlib.reload(llm_clients)
        from llm_clients import get_llm_client
        from langchain.schema import HumanMessage, SystemMessage

        client = get_llm_client("minimax", "MiniMax-M2.7-highspeed")
        messages = [
            SystemMessage(content="You are a helpful assistant. Reply in one sentence."),
            HumanMessage(content="What is 2+2?"),
        ]
        response = client.invoke(messages)
        assert isinstance(response, str)
        assert len(response) > 0
        assert "4" in response

    def test_minimax_live_get_model_response(self):
        """Integration: get_model_response() with MiniMax."""
        import importlib
        import llm_clients

        importlib.reload(llm_clients)
        from llm_clients import get_llm_client, get_model_response

        client = get_llm_client("minimax", "MiniMax-M2.7-highspeed")
        response = get_model_response(
            client,
            system_prompt="You are helpful. Reply in one sentence.",
            user_prompt="What color is the sky on a clear day?",
        )
        assert isinstance(response, str)
        assert len(response) > 0

    def test_minimax_live_streaming(self):
        """Integration: MiniMax streaming should return complete response."""
        import importlib
        import llm_clients

        importlib.reload(llm_clients)
        from llm_clients import get_llm_client
        from langchain.schema import HumanMessage, SystemMessage

        client = get_llm_client("minimax")
        messages = [
            SystemMessage(content="Reply with exactly: Hello World"),
            HumanMessage(content="Say the phrase"),
        ]
        response = client.invoke(messages)
        assert isinstance(response, str)
        assert len(response) > 0
