from http import HTTPStatus
from typing import Annotated, Optional, Union

import sqlalchemy as sa
from fastapi import APIRouter, Body, Depends, HTTPException, UploadFile, status
from langflow_base.api.v1.schemas import (
    CustomComponentCode,
    ProcessResponse,
    TaskResponse,
    TaskStatusResponse,
    UploadFileResponse,
)
from langflow_base.interface.custom.custom_component import CustomComponent
from langflow_base.interface.custom.directory_reader import DirectoryReader
from langflow_base.interface.types import build_langchain_template_custom_component, create_and_validate_component
from langflow_base.processing.process import process_graph_cached, process_tweaks
from langflow_base.services.auth.utils import api_key_security, get_current_active_user
from langflow_base.services.cache.utils import save_uploaded_file
from langflow_base.services.database.models.flow import Flow
from langflow_base.services.database.models.user.model import User
from langflow_base.services.deps import get_session, get_session_service, get_settings_service, get_task_service
from loguru import logger

try:
    from langflow_base.worker import process_graph_cached_task
except ImportError:

    def process_graph_cached_task(*args, **kwargs):
        raise NotImplementedError("Celery is not installed")


from langflow_base.services.task.service import TaskService
from sqlmodel import Session

# build router
router = APIRouter(tags=["Base"])


@router.get("/all", dependencies=[Depends(get_current_active_user)])
def get_all(
    settings_service=Depends(get_settings_service),
):
    from langflow_base.interface.types import get_all_types_dict

    logger.debug("Building langchain types dict")
    try:
        return get_all_types_dict(settings_service)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# For backwards compatibility we will keep the old endpoint
@router.post(
    "/predict/{flow_id}",
    response_model=ProcessResponse,
    dependencies=[Depends(api_key_security)],
)
@router.post(
    "/process/{flow_id}",
    response_model=ProcessResponse,
)
async def process(
    session: Annotated[Session, Depends(get_session)],
    flow_id: str,
    inputs: Optional[dict] = None,
    tweaks: Optional[dict] = None,
    clear_cache: Annotated[bool, Body(embed=True)] = False,  # noqa: F821
    session_id: Annotated[Union[None, str], Body(embed=True)] = None,  # noqa: F821
    task_service: "TaskService" = Depends(get_task_service),
    api_key_user: User = Depends(api_key_security),
    sync: Annotated[bool, Body(embed=True)] = True,  # noqa: F821
):
    """
    Endpoint to process an input with a given flow_id.
    """

    try:
        if api_key_user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key",
            )

        # Get the flow that matches the flow_id and belongs to the user
        flow = session.query(Flow).filter(Flow.id == flow_id).filter(Flow.user_id == api_key_user.id).first()
        if flow is None:
            raise ValueError(f"Flow {flow_id} not found")

        if flow.data is None:
            raise ValueError(f"Flow {flow_id} has no data")
        graph_data = flow.data
        task_result = None
        if tweaks:
            try:
                graph_data = process_tweaks(graph_data, tweaks)
            except Exception as exc:
                logger.error(f"Error processing tweaks: {exc}")
        if sync:
            task_id, result = await task_service.launch_and_await_task(
                process_graph_cached_task if task_service.use_celery else process_graph_cached,
                graph_data,
                inputs,
                clear_cache,
                session_id,
            )
            if isinstance(result, dict) and "result" in result:
                task_result = result["result"]
                session_id = result["session_id"]
            elif hasattr(result, "result") and hasattr(result, "session_id"):
                task_result = result.result

                session_id = result.session_id
        else:
            logger.warning(
                "This is an experimental feature and may not work as expected."
                "Please report any issues to our GitHub repository."
            )
            if session_id is None:
                # Generate a session ID
                session_id = get_session_service().generate_key(session_id=session_id, data_graph=graph_data)
            task_id, task = await task_service.launch_task(
                process_graph_cached_task if task_service.use_celery else process_graph_cached,
                graph_data,
                inputs,
                clear_cache,
                session_id,
            )
            task_result = task.status

        if task_id:
            task_response = TaskResponse(id=task_id, href=f"api/v1/task/{task_id}")
        else:
            task_response = None

        return ProcessResponse(
            result=task_result,
            task=task_response,
            session_id=session_id,
            backend=task_service.backend_name,
        )
    except sa.exc.StatementError as exc:
        # StatementError('(builtins.ValueError) badly formed hexadecimal UUID string')
        if "badly formed hexadecimal UUID string" in str(exc):
            # This means the Flow ID is not a valid UUID which means it can't find the flow
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        if f"Flow {flow_id} not found" in str(exc):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        else:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except Exception as e:
        # Log stack trace
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    task_service = get_task_service()
    task = task_service.get_task(task_id)
    result = None
    if task.ready():
        result = task.result
        if isinstance(result, dict) and "result" in result:
            result = result["result"]
        elif hasattr(result, "result"):
            result = result.result

    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatusResponse(status=task.status, result=result)


@router.post(
    "/upload/{flow_id}",
    response_model=UploadFileResponse,
    status_code=HTTPStatus.CREATED,
)
async def create_upload_file(file: UploadFile, flow_id: str):
    # Cache file
    try:
        file_path = save_uploaded_file(file, folder_name=flow_id)

        return UploadFileResponse(
            flowId=flow_id,
            file_path=file_path,
        )
    except Exception as exc:
        logger.error(f"Error saving file: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# get endpoint to return version of langflow
@router.get("/version")
def get_version():
    from langflow import __version__

    return {"version": __version__}


@router.post("/custom_component", status_code=HTTPStatus.OK)
async def custom_component(
    raw_code: CustomComponentCode,
    user: User = Depends(get_current_active_user),
):
    create_and_validate_component(raw_code.code)

    extractor = CustomComponent(code=raw_code.code)
    extractor.validate()

    return build_langchain_template_custom_component(extractor, user_id=user.id)


@router.post("/custom_component/reload", status_code=HTTPStatus.OK)
async def reload_custom_component(path: str, user: User = Depends(get_current_active_user)):
    from langflow_base.interface.types import build_langchain_template_custom_component

    try:
        reader = DirectoryReader("")
        valid, content = reader.process_file(path)
        if not valid:
            raise ValueError(content)

        extractor = CustomComponent(code=content)
        extractor.validate()
        return build_langchain_template_custom_component(extractor, user_id=user.id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/custom_component/update", status_code=HTTPStatus.OK)
async def custom_component_update(
    raw_code: CustomComponentCode,
    user: User = Depends(get_current_active_user),
):
    component = create_and_validate_component(raw_code.code)

    component_node = build_langchain_template_custom_component(component, user_id=user.id, update_field=raw_code.field)
    # Update the field
    return component_node