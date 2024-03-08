import os
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pytest
import respx
from httpx import Response

from langflow.components import (
    data,
)  # Adjust the import according to your project structure


@pytest.fixture
def api_request():
    # This fixture provides an instance of APIRequest for each test case
    return data.APIRequest()


@pytest.mark.asyncio
@respx.mock
async def test_successful_get_request(api_request):
    # Mocking a successful GET request
    url = "https://example.com/api/test"
    method = "GET"
    mock_response = {"success": True}
    respx.get(url).mock(return_value=Response(200, json=mock_response))

    # Making the request
    result = await api_request.make_request(
        client=httpx.AsyncClient(), method=method, url=url
    )

    # Assertions
    assert result.data["status_code"] == 200
    assert result.data["result"] == mock_response


@pytest.mark.asyncio
@respx.mock
async def test_failed_request(api_request):
    # Mocking a failed GET request
    url = "https://example.com/api/test"
    method = "GET"
    respx.get(url).mock(return_value=Response(404))

    # Making the request
    result = await api_request.make_request(
        client=httpx.AsyncClient(), method=method, url=url
    )

    # Assertions
    assert result.data["status_code"] == 404


@pytest.mark.asyncio
@respx.mock
async def test_timeout(api_request):
    # Mocking a timeout
    url = "https://example.com/api/timeout"
    method = "GET"
    respx.get(url).mock(
        side_effect=httpx.TimeoutException(message="Timeout", request=None)
    )

    # Making the request
    result = await api_request.make_request(
        client=httpx.AsyncClient(), method=method, url=url, timeout=1
    )

    # Assertions
    assert result.data["status_code"] == 408
    assert result.data["error"] == "Request timed out"


@pytest.mark.asyncio
@respx.mock
async def test_build_with_multiple_urls(api_request):
    # This test depends on having a working internet connection and accessible URLs
    # It's better to mock these requests using respx or a similar library

    # Setup for multiple URLs
    method = "GET"
    urls = ["https://example.com/api/one", "https://example.com/api/two"]
    # You would mock these requests similarly to the single request tests
    for url in urls:
        respx.get(url).mock(return_value=Response(200, json={"success": True}))

    # Do I have to mock the async client?
    #

    # Execute the build method
    results = await api_request.build(method=method, urls=urls)

    # Assertions
    assert len(results) == len(urls)


@patch("langflow.components.data.Directory.parallel_load_records")
@patch("langflow.components.data.Directory.retrieve_file_paths")
@patch("langflow.components.data.DirectoryComponent.resolve_path")
def test_directory_component_build_with_multithreading(
    mock_resolve_path, mock_retrieve_file_paths, mock_parallel_load_records
):
    # Arrange
    directory_component = data.DirectoryComponent()
    path = os.path.dirname(os.path.abspath(__file__))
    types = ["py"]
    depth = 1
    max_concurrency = 2
    load_hidden = False
    recursive = True
    silent_errors = False
    use_multithreading = True

    mock_resolve_path.return_value = path
    mock_retrieve_file_paths.return_value = [
        os.path.join(path, file) for file in os.listdir(path) if file.endswith(".py")
    ]
    mock_parallel_load_records.return_value = [Mock()]

    # Act
    result = directory_component.build(
        path,
        types,
        depth,
        max_concurrency,
        load_hidden,
        recursive,
        silent_errors,
        use_multithreading,
    )

    # Assert
    mock_resolve_path.assert_called_once_with(path)
    mock_retrieve_file_paths.assert_called_once_with(
        path, types, load_hidden, recursive, depth
    )
    mock_parallel_load_records.assert_called_once_with(
        mock_retrieve_file_paths.return_value, silent_errors, max_concurrency
    )


def test_directory_without_mocks():
    directory_component = data.DirectoryComponent()
    from langflow.initial_setup import setup
    from langflow.initial_setup.setup import load_starter_projects

    projects = load_starter_projects()
    # the setup module has a folder where the projects are stored
    # the contents of that folder are in the projects variable
    # the directory component can be used to load the projects
    # and we can validate if the contents are the same as the projects variable
    setup_path = Path(setup.__file__).parent / "starter_projects"
    result = directory_component.build(
        str(setup_path), types=["json"], use_multithreading=False
    )
    assert len(result) == len(projects)