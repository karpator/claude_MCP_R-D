from . import CommonLanguage


def status(status_code: CommonLanguage):
    return {
        "status__": f"{status_code.value}\n\n",
    }
