from fastapi import APIRouter


router = APIRouter()


@router.get('/system-info',tags=['web'])
def system_info():
    raise NotImplementedError
