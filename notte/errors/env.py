from notte.errors.base import NotteBaseError, UnexpectedBehaviorError


class NoContextObservedError(UnexpectedBehaviorError):
    def __init__(self) -> None:
        super().__init__(
            message="tried to access `env.context` but no context is available in the environment",
            advice="You should use `await env.observe()` first to get a context",
        )


class MaxStepsReachedError(NotteBaseError):
    def __init__(self, max_steps: int) -> None:
        super().__init__(
            dev_message=(
                f"Max number steps reached: {max_steps} in the currrent trajectory. Either use "
                "`env.reset()` to reset the environment or increase max steps in `NotteEnv(max_steps=..)`."
            ),
            user_message=(
                f"Too many actions executed in the current session (i.e. {max_steps} actions). "
                "Please start a new session to continue."
            ),
        )
