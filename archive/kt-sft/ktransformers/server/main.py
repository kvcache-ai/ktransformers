import os
import re
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn.logging
import uvicorn
import sys
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
from fastapi.middleware.cors import CORSMiddleware
from ktransformers.server.args import ArgumentParser
from ktransformers.server.config.config import Config
from ktransformers.server.utils.create_interface import create_interface, GlobalInterface
from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ktransformers.server.api import router, post_db_creation_operations
from ktransformers.server.utils.sql_utils import Base, SQLUtil
from ktransformers.server.config.log import logger


def mount_app_routes(mount_app: FastAPI):
    sql_util = SQLUtil()
    logger.info("Creating SQL tables")
    Base.metadata.create_all(bind=sql_util.sqlalchemy_engine)
    post_db_creation_operations()
    mount_app.include_router(router)


def create_app():
    cfg = Config()
    if(hasattr(GlobalInterface.interface, "lifespan")):
        app = FastAPI(lifespan=GlobalInterface.interface.lifespan)
    else:
        app = FastAPI()
    if Config().web_cross_domain:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    mount_app_routes(app)
    if cfg.mount_web:
        mount_index_routes(app)
    return app


def update_web_port(config_file: str):
    ip_port_pattern = (
        r"(localhost|((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)):[0-9]{1,5}"
    )
    with open(config_file, "r", encoding="utf-8") as f_cfg:
        web_config = f_cfg.read()
    ip_port = "localhost:" + str(Config().server_port)
    new_web_config = re.sub(ip_port_pattern, ip_port, web_config)
    with open(config_file, "w", encoding="utf-8") as f_cfg:
        f_cfg.write(new_web_config)


def mount_index_routes(app: FastAPI):
    project_dir = os.path.dirname(os.path.dirname(__file__))
    web_dir = os.path.join(project_dir, "website/dist")
    web_config_file = os.path.join(web_dir, "config.js")
    update_web_port(web_config_file)
    if os.path.exists(web_dir):
        app.mount("/web", StaticFiles(directory=web_dir), name="static")
    else:
        err_str = f"No website resources in {web_dir}, please complile the website by npm first"
        logger.error(err_str)
        print(err_str)
        exit(1)


def run_api(app, host, port, **kwargs):
    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        uvicorn.run(
            app,
            host=host,
            port=port,
            ssl_keyfile=kwargs.get("ssl_keyfile"),
            ssl_certfile=kwargs.get("ssl_certfile"),
        )
    else:
        uvicorn.run(app, host=host, port=port, log_level="debug")


def custom_openapi(app):
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="ktransformers server",
        version="1.0.0",
        summary="This is a server that provides a RESTful API for ktransformers.",
        description="We provided chat completion and openai assistant interfaces.",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {"url": "https://kvcache.ai/media/icon_1.png"}
    app.openapi_schema = openapi_schema
    return app.openapi_schema


def main():
    cfg = Config()

    arg_parser = ArgumentParser(cfg)

    args = arg_parser.parse_args()
    create_interface(config=cfg, default_args=cfg)
    app = create_app()
    custom_openapi(app)

    run_api(
        app=app,
        host=args.host,
        port=args.port,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )

if __name__ == "__main__":
    main()
