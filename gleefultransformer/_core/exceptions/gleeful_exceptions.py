class GleefulChatError(Exception):
    pass


class GleefulInputError(GleefulChatError):
    pass


class GleefulLoadModelError(GleefulChatError):
    pass


class GleefulLoadTokenizerError(GleefulChatError):
    pass


class GleefulResumeError(GleefulChatError):
    pass


class GleefulInferenceError(GleefulChatError):
    pass


class GleefulDeviceError(GleefulChatError):
    pass


class GleefulIOError(GleefulChatError):
    pass


class GleefulTokenizationError(GleefulChatError):
    pass


class GleefulDatasetError(GleefulChatError):
    pass


class GleefulSaveError(GleefulChatError):
    pass


class GleefulNoModelError(GleefulChatError):
    pass


class GleefulPushError(GleefulChatError):
    pass


class GleefulPlotError(GleefulChatError):
    pass
