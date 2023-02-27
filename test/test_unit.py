import json
from pathlib import Path

from steamship import Block, File, TaskState
from steamship.data import GenerationTag, TagKind, TagValueKey
from steamship.plugin.inputs.block_and_tag_plugin_input import BlockAndTagPluginInput
from steamship.plugin.request import PluginRequest

from src.api import CerebriumPlugin


def test_tagger():
    config = json.load(Path("config.json").open())
    tagger = CerebriumPlugin(config=config)
    content = Path("data/roses.txt").open().read()
    file = File(id="foo", blocks=[Block(text=content, tags=[])])
    request = PluginRequest(data=BlockAndTagPluginInput(file=file))
    response = tagger.run(request)

    assert response.status.state is TaskState.succeeded
    assert response.data is not None
    assert response.data.file is not None

    file = response.data.file

    assert file.tags is not None
    assert len(file.tags) == 1

    tags = file.blocks[0].tags
    for tag in tags:
        assert tag.kind == TagKind.GENERATION
        assert tag.name == GenerationTag.PROMPT_COMPLETION
        tag_value = tag.value[TagValueKey.STRING_VALUE]
        assert tag_value is not None
        assert isinstance(tag_value, str)


def test_tagger_multiblock():
    config = json.load(Path("config.json").open())
    tagger = CerebriumPlugin(config=config)
    file = File(id="foo",
                blocks=[Block(text="Let's count: one two three"), Block(text="The primary colors are: red blue")])
    request = PluginRequest(data=BlockAndTagPluginInput(file=file))
    response = tagger.run(request)

    assert response.status.state is TaskState.succeeded
    assert response.data is not None
    assert response.data.file is not None
    assert response.data.file.blocks is not None
