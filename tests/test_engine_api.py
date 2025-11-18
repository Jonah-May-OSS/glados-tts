"""Tests for the Flask engine API."""

import pytest
import sys
import os
from pathlib import Path


# Mark all tests as requiring models
pytestmark = pytest.mark.skipif(
    not Path("models/glados-new.pt").exists(), reason="Models not available"
)


class TestEngineAPI:
    """Tests for the remote TTS engine API."""

    @pytest.fixture
    def app(self):
        """Create a test Flask app instance."""
        # Import needs to be done here to avoid issues when models aren't available
        sys.path.insert(0, os.getcwd() + "/glados_tts")
        # We need to mock or carefully import the engine module
        # For now, we'll skip the actual import and test what we can
        pytest.skip("Engine API tests require careful mocking - implement in future")

    def test_synthesize_endpoint_basic(self, app):
        """Test basic synthesize endpoint."""
        # This would test the /synthesize/<text> endpoint
        pass

    def test_synthesize_endpoint_empty(self, app):
        """Test synthesize endpoint with empty text."""
        # Should return "No input"
        pass

    def test_synthesize_caching(self, app):
        """Test that audio caching works correctly."""
        # Test that repeated requests for same text use cache
        pass
