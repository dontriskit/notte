import pytest

from notte.actions.base import BrowserAction
from notte.controller.actions import BrowserActionId
from notte.env import NotteEnv
from tests.mock.mock_service import MockLLMService


@pytest.fixture
def llm_service():
    return MockLLMService(mock_response="<data-extraction> # Hello World </data-extraction>")


def test_browser_actions_list():
    """Test that all special actions are properly defined"""
    browser_actions = BrowserAction.list()

    # Test we have all 8 special actions
    assert len(browser_actions) == len(BrowserActionId)

    # Test each special action ID exists
    action_ids = set([action.id for action in browser_actions])
    expected_ids = set(BrowserActionId)
    assert action_ids == expected_ids

    # Test special action detection
    for action_id in expected_ids:
        assert BrowserAction.is_special(action_id)

    # Test non-special action detection
    assert not BrowserAction.is_special("B1")
    assert not BrowserAction.is_special("I1")
    assert not BrowserAction.is_special("L1")


@pytest.mark.asyncio
async def test_goto_and_scrape(llm_service: MockLLMService):
    """Test the execution of various special actions"""
    async with NotteEnv(headless=True, llmserve=llm_service) as env:
        # Test S1: Go to URL
        obs = await env.execute(action_id=BrowserActionId.GOTO, params={"url": "https://github.com/"})
        assert obs.clean_url == "github.com"

        # Test S2: Scrape data
        obs = await env.execute(action_id=BrowserActionId.SCRAPE)
        assert obs.data is not None
        assert obs.data.markdown == "# Hello World"


@pytest.mark.asyncio
async def test_go_back_and_forward(llm_service: MockLLMService):
    """Test the execution of various special actions"""
    async with NotteEnv(headless=True, llmserve=llm_service) as env:
        # Test S4: Go to notte
        obs = await env.execute(action_id=BrowserActionId.GOTO, params={"url": "https://github.com/"})
        assert obs.clean_url == "github.com"
        # Test S4: Go back
        obs = await env.execute(action_id=BrowserActionId.GOTO, params={"url": "https://google.com/"})
        assert obs.clean_url == "google.com"
        obs = await env.execute(action_id=BrowserActionId.GO_BACK)
        assert obs.clean_url == "github.com"

        # Test S5: Go forward
        obs = await env.execute(action_id=BrowserActionId.GO_FORWARD)
        assert obs.clean_url == "google.com"


@pytest.mark.asyncio
async def test_wait_and_complete(llm_service: MockLLMService):
    """Test the execution of various special actions"""
    async with NotteEnv(headless=True, llmserve=llm_service) as env:

        # Test S4: Go goto goole
        obs = await env.execute(action_id=BrowserActionId.GOTO, params={"url": "https://google.com/"})
        assert obs.clean_url == "google.com"

        # Test S7: Wait
        _ = await env.execute(action_id=BrowserActionId.WAIT, params={"value": "1"})

        # Test S8: Terminate session (cannot execute any actions after this)
        _ = await env.execute(action_id=BrowserActionId.COMPLETION, params={"success": "true", "answer": "Hello World"})
        with pytest.raises(ValueError, match="Browser not started"):
            _ = await env.goto("https://github.com/")


@pytest.mark.asyncio
async def test_special_action_validation(llm_service: MockLLMService):
    """Test validation of special action parameters"""
    async with NotteEnv(headless=True, llmserve=llm_service) as env:
        _ = await env.goto("https://github.com/")
        # Test S1 requires URL parameter
        with pytest.raises(ValueError, match=f"Action with id '{BrowserActionId.GOTO}' is invalid"):
            _ = await env.execute(action_id=BrowserActionId.GOTO)

        # Test S7 requires wait time parameter
        with pytest.raises(ValueError, match=f"Action with id '{BrowserActionId.WAIT}' is invalid"):
            _ = await env.execute(action_id=BrowserActionId.WAIT)

        # Test invalid special action
        with pytest.raises(ValueError, match="X1 not found"):
            _ = await env.execute("X1")
