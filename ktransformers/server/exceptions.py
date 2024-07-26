from fastapi import HTTPException, status


def db_exception():
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="DB Error",
    )


def not_implemented(what):
    return HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=f"{what} not implemented",
    )


def internal_server_error(what):
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"{what}")


def request_error(what):
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{what}")
