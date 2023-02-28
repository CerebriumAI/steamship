"""Default generation plugin for prompts."""
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Type
import json
import requests
from cerebrium import model_api_request
from pydantic import Field
from steamship import Steamship, Tag, SteamshipError
from steamship.data import GenerationTag, TagKind, TagValueKey
from steamship.invocable import Config, InvocableResponse, InvocationContext
from steamship.plugin.inputs.block_and_tag_plugin_input import BlockAndTagPluginInput
from steamship.plugin.outputs.block_and_tag_plugin_output import BlockAndTagPluginOutput
from steamship.plugin.request import PluginRequest
from steamship.plugin.tagger import Tagger
from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential, before_sleep_log, wait_random_exponential, wait_exponential_jitter,
)



class CerebriumPlugin(Tagger):
    """Default plugin for generating text based on a prompt.

    Notes
    -----
    * The parameters logit_bias, logprobs and stream are not supported.
    """

    class CerebriumPluginConfig(Config):
        cerebrium_api_key: str = Field("",
                                    description="A Cerebrium API key to use. If left default, will use Steamship's API key.")
        endpoint: str = Field(description="The URL endpoint of your deployed model on Cerebrium.  Can be a pre-existing or fine-tuned model.")
        webhook_endpoint: Optional[str] = Field(description="The url endpoint you would like us to send the model results to once it has finished.")
        
        max_retries: int = Field(1, description="Maximum number of retries to make when generating.")
        request_timeout: Optional[float] = Field(600,
                                                 description="Timeout for requests to Cerebrium completion API. Default is 600 seconds.")

        #Image models
        height: Optional[int] = Field(512, description="The height of the image generation")
        width: Optional[int] = Field(512, description="The width of the image generation")
        num_inference_steps: Optional[int] = Field(50, description="The number of steps you would like the model to take to generate the image.")
        guidance_scale: Optional[int] = Field(8, description="A way to increase the adherence to the conditional signal that guides the generation")
        num_images_per_prompt: Optional[int] = Field(1, description="The number of image variations you would like the model to generate.")
        negative_prompt: Optional[str] = Field(description="The negative prompt is a parameter that tells the model what you don’t want to see in the generated images")
        image: Optional[str] = Field(description="This is a base64 encoded string of your initial image.")
        hf_token: Optional[str] = Field(description="This is the token from your HuggingFace profile in order to access your model repo.")
        model_id: Optional[str] = Field(description="This is the Hugging Face id of your model repo.")
        

        #Language Models
        audio: Optional[str] = Field(description="A base64 encoded string of the audio file you would like to transcribe/translate.")
        max_length: Optional[int] = Field(description="The maximum number of words to generate per request")
        temperature: Optional[float] = Field(0.4,
                                             description="Controls randomness. Lower values produce higher likelihood / more predictable results; higher values produce more variety. Values between 0-1.")
        echo: Optional[bool] = Field(False, description="Echo back the prompt in addition to the completion")
    
    @classmethod
    def config_cls(cls) -> Type[Config]:
        return cls.CerebriumPluginConfig

    config: CerebriumPluginConfig

    def __init__(
            self,
            client: Steamship = None,
            config: Dict[str, Any] = None,
            context: InvocationContext = None,
    ):
        super().__init__(client, config, context)


    def generation_with_retry(self, **kwargs: Any) -> Any:
        """Use tenacity to retry the generation call."""

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential_jitter(jitter=5),
            before_sleep=before_sleep_log(logging.root, logging.INFO),
            after=after_log(logging.root, logging.INFO),
        )
        def _generation_with_retry(**kwargs: Any) -> Any:
            headers = {"Authorization": self.config.cerebrium_api_key, "Content-Type": "application/json"}
            response = requests.request(
                "POST", self.config.endpoint, headers=headers, data=json.dumps({"prompt": kwargs.get('prompt')}), timeout=30
            )
            result = json.loads(response.text)
            return result

        result =  _generation_with_retry(**kwargs)
        logging.info("Retry statistics: "+ json.dumps(_generation_with_retry.retry.statistics))
        return result

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {
            "echo": self.config.echo,
            "request_timeout": self.config.request_timeout

        }

    def _generate(
            self, prompt: str,
            user: Optional[str] = None
    ) -> (List[List[str]], Dict[str, int]):
    
        logging.info(f"Making Cerebrium generation call on behalf of user with id: {user}")
        """Call the API to generate response"""

        invocation_params = {
            "prompt": prompt,
            "user": user or "",
            **self._default_params  # Note: stops are not invocation params just yet
        }
        generation = self.generation_with_retry(
            **invocation_params
        )
        
        return generation

    def run(
             self, request: PluginRequest[BlockAndTagPluginInput]
    ) -> InvocableResponse[BlockAndTagPluginOutput]:
        """Run the text generator against all Blocks of text."""
        file = request.data.file
        for block in file.blocks:
            generated_text = self._generate(prompt=block.text)
            output = generated_text.get('result', None)
            if output:
                block.tags.append(
                    Tag(
                        kind=TagKind.GENERATION,
                        name=GenerationTag.PROMPT_COMPLETION,
                        value={TagValueKey.STRING_VALUE: output},
                    )
                )

        return InvocableResponse(data=BlockAndTagPluginOutput(file=file))