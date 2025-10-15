# `smolagents`

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/license_to_call.png" style="max-width:700px"/>
</div>

## What is smolagents?

`smolagents` is an open-source Python library designed to make it extremely easy to build and run agents using just a few lines of code.

Key features of `smolagents` include:

✨ **Simplicity**: The logic for agents fits in ~thousand lines of code. We kept abstractions to their minimal shape above raw code!

🧑‍💻 **First-class support for Code Agents**: [`CodeAgent`](reference/agents#smolagents.CodeAgent) writes its actions in code (as opposed to "agents being used to write code") to invoke tools or perform computations, enabling natural composability (function nesting, loops, conditionals). To make it secure, we support [executing in sandboxed environment](tutorials/secure_code_execution) via [E2B](https://e2b.dev/) or via Docker.

📡 **Common Tool-Calling Agent Support**: In addition to CodeAgents, [`ToolCallingAgent`](reference/agents#smolagents.ToolCallingAgent) supports usual JSON/text-based tool-calling for scenarios where that paradigm is preferred.

🤗 **Hub integrations**: Seamlessly share and load agents and tools to/from the Hub as Gradio Spaces.

🌐 **Model-agnostic**: Easily integrate any large language model (LLM), whether it's hosted on the Hub via [Inference providers](https://huggingface.co/docs/inference-providers/index), accessed via APIs such as OpenAI, Anthropic, or many others via LiteLLM integration, or run locally using Transformers or Ollama. Powering an agent with your preferred LLM is straightforward and flexible.

👁️ **Modality-agnostic**: Beyond text, agents can handle vision, video, and audio inputs, broadening the range of possible applications. Check out [this tutorial](examples/web_browser) for vision.

🛠️ **Tool-agnostic**: You can use tools from any [MCP server](reference/tools#smolagents.ToolCollection.from_mcp), from [LangChain](reference/tools#smolagents.Tool.from_langchain), you can even use a [Hub Space](reference/tools#smolagents.Tool.from_space) as a tool.

💻 **CLI Tools**: Comes with command-line utilities (smolagent, webagent) for quickly running agents without writing boilerplate code.

## Quickstart

<DocNotebookDropdown
  classNames="absolute z-10 right-0 top-0"
  options={[
    {label: "Mixed", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/main/smolagents_doc/en/index.ipynb"},
    {label: "PyTorch", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/main/smolagents_doc/en/pytorch/index.ipynb"},
    {label: "TensorFlow", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/main/smolagents_doc/en/tensorflow/index.ipynb"},
    {label: "Mixed", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/smolagents_doc/en/index.ipynb"},
    {label: "PyTorch", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/smolagents_doc/en/pytorch/index.ipynb"},
    {label: "TensorFlow", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/smolagents_doc/en/tensorflow/index.ipynb"},
]} />

Get started with smolagents in just a few minutes! This guide will show you how to create and run your first agent.

### Installation

Install smolagents with pip:

```bash
pip install 'smolagents[toolkit]'  # Includes default tools like web search
```

### Create Your First Agent

Here's a minimal example to create and run an agent:

```python
from smolagents import CodeAgent, InferenceClientModel

# Initialize a model (using Hugging Face Inference API)
model = InferenceClientModel()  # Uses a default model

# Create an agent with no tools
agent = CodeAgent(tools=[], model=model)

# Run the agent with a task
result = agent.run("Calculate the sum of numbers from 1 to 10")
print(result)
```

That's it! Your agent will use Python code to solve the task and return the result.

### Adding Tools

Let's make our agent more capable by adding some tools:

```python
from smolagents import CodeAgent, InferenceClientModel, DuckDuckGoSearchTool

model = InferenceClientModel()
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
)

# Now the agent can search the web!
result = agent.run("What is the current weather in Paris?")
print(result)
```

### Using Different Models

You can use various models with your agent:

```python
# Using a specific model from Hugging Face
model = InferenceClientModel(model_id="meta-llama/Llama-2-70b-chat-hf")

# Using OpenAI/Anthropic (requires 'smolagents[litellm]')
from smolagents import LiteLLMModel
model = LiteLLMModel(model_id="gpt-4")

# Using local models (requires 'smolagents[transformers]')
from smolagents import TransformersModel
model = TransformersModel(model_id="meta-llama/Llama-2-7b-chat-hf")
```

## Next Steps

- Learn how to set up smolagents with various models and tools in the [Installation Guide](installation)
- Check out the [Guided Tour](guided_tour) for more advanced features
- Learn about [building custom tools](tutorials/tools)
- Explore [secure code execution](tutorials/secure_code_execution)
- See how to create [multi-agent systems](tutorials/building_good_agents)

<div class="mt-10">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-2 md:gap-y-4 md:gap-x-5">
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./guided_tour"
      ><div class="w-full text-center bg-gradient-to-br from-blue-400 to-blue-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">Guided tour</div>
      <p class="text-gray-700">Learn the basics and become familiar with using Agents. Start here if you are using Agents for the first time!</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./examples/text_to_sql"
      ><div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">How-to guides</div>
      <p class="text-gray-700">Practical guides to help you achieve a specific goal: create an agent to generate and test SQL queries!</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./conceptual_guides/intro_agents"
      ><div class="w-full text-center bg-gradient-to-br from-pink-400 to-pink-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">Conceptual guides</div>
      <p class="text-gray-700">High-level explanations for building a better understanding of important topics.</p>
   </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./tutorials/building_good_agents"
      ><div class="w-full text-center bg-gradient-to-br from-purple-400 to-purple-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">Tutorials</div>
      <p class="text-gray-700">Horizontal tutorials that cover important aspects of building agents.</p>
    </a>
  </div>
</div>


<EditOnGithub source="https://github.com/huggingface/smolagents/blob/main/docs/source/en/index.md" />



# Orchestrate a multi-agent system 🤖🤝🤖

<DocNotebookDropdown
  classNames="absolute z-10 right-0 top-0"
  options={[
    {label: "Mixed", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/main/smolagents_doc/en/multiagents.ipynb"},
    {label: "PyTorch", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/main/smolagents_doc/en/pytorch/multiagents.ipynb"},
    {label: "TensorFlow", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/main/smolagents_doc/en/tensorflow/multiagents.ipynb"},
    {label: "Mixed", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/smolagents_doc/en/multiagents.ipynb"},
    {label: "PyTorch", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/smolagents_doc/en/pytorch/multiagents.ipynb"},
    {label: "TensorFlow", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/smolagents_doc/en/tensorflow/multiagents.ipynb"},
]} />

In this notebook we will make a **multi-agent web browser: an agentic system with several agents collaborating to solve problems using the web!**

It will be a simple hierarchy:

```
              +----------------+
              | Manager agent  |
              +----------------+
                       |
        _______________|______________
       |                              |
Code Interpreter            +------------------+
    tool                    | Web Search agent |
                            +------------------+
                               |            |
                        Web Search tool     |
                                   Visit webpage tool
```
Let's set up this system. 

Run the line below to install the required dependencies:

```py
!pip install 'smolagents[toolkit]' --upgrade -q
```

Let's login to HF in order to call Inference Providers:

```py
from huggingface_hub import login

login()
```

⚡️ Our agent will be powered by [Qwen/Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) using `InferenceClientModel` class that uses HF's Inference API: the Inference API allows to quickly and easily run any OS model.

> [!TIP]
> Inference Providers give access to hundreds of models, powered by serverless inference partners. A list of supported providers can be found [here](https://huggingface.co/docs/inference-providers/index).

```py
model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
```

## 🔍 Create a web search tool

For web browsing, we can already use our native [WebSearchTool](/docs/smolagents/v1.22.0/en/reference/default_tools#smolagents.WebSearchTool) tool to provide a Google search equivalent.

But then we will also need to be able to peak into the page found by the `WebSearchTool`.
To do so, we could import the library's built-in `VisitWebpageTool`, but we will build it again to see how it's done.

So let's create our `VisitWebpageTool` tool from scratch using `markdownify`.

```py
import re
import requests
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import tool


@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Convert the HTML content to Markdown
        markdown_content = markdownify(response.text).strip()

        # Remove multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        return markdown_content

    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
```

Ok, now let's initialize and test our tool!

```py
print(visit_webpage("https://en.wikipedia.org/wiki/Hugging_Face")[:500])
```

## Build our multi-agent system 🤖🤝🤖

Now that we have all the tools `search` and `visit_webpage`, we can use them to create the web agent.

Which configuration to choose for this agent?
- Web browsing is a single-timeline task that does not require parallel tool calls, so JSON tool calling works well for that. We thus choose a `ToolCallingAgent`.
- Also, since sometimes web search requires exploring many pages before finding the correct answer, we prefer to increase the number of `max_steps` to 10.

```py
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    InferenceClientModel,
    WebSearchTool,
    LiteLLMModel,
)

model = InferenceClientModel(model_id=model_id)

web_agent = ToolCallingAgent(
    tools=[WebSearchTool(), visit_webpage],
    model=model,
    max_steps=10,
    name="web_search_agent",
    description="Runs web searches for you.",
)
```

Note that we gave this agent attributes `name` and `description`, mandatory attributes to make this agent callable by its manager agent.

Then we create a manager agent, and upon initialization we pass our managed agent to it in its `managed_agents` argument.

Since this agent is the one tasked with the planning and thinking, advanced reasoning will be beneficial, so a `CodeAgent` will work well.

Also, we want to ask a question that involves the current year and does additional data calculations: so let us add `additional_authorized_imports=["time", "numpy", "pandas"]`, just in case the agent needs these packages.

```py
manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[web_agent],
    additional_authorized_imports=["time", "numpy", "pandas"],
)
```

That's all! Now let's run our system! We select a question that requires both some calculation and research:

```py
answer = manager_agent.run("If LLM training continues to scale up at the current rhythm until 2030, what would be the electric power in GW required to power the biggest training runs by 2030? What would that correspond to, compared to some countries? Please provide a source for any numbers used.")
```

We get this report as the answer:
```
Based on current growth projections and energy consumption estimates, if LLM trainings continue to scale up at the 
current rhythm until 2030:

1. The electric power required to power the biggest training runs by 2030 would be approximately 303.74 GW, which 
translates to about 2,660,762 GWh/year.

2. Comparing this to countries' electricity consumption:
   - It would be equivalent to about 34% of China's total electricity consumption.
   - It would exceed the total electricity consumption of India (184%), Russia (267%), and Japan (291%).
   - It would be nearly 9 times the electricity consumption of countries like Italy or Mexico.

3. Source of numbers:
   - The initial estimate of 5 GW for future LLM training comes from AWS CEO Matt Garman.
   - The growth projection used a CAGR of 79.80% from market research by Springs.
   - Country electricity consumption data is from the U.S. Energy Information Administration, primarily for the year 
2021.
```

Seems like we'll need some sizeable powerplants if the [scaling hypothesis](https://gwern.net/scaling-hypothesis) continues to hold true.

Our agents managed to efficiently collaborate towards solving the task! ✅

💡 You can easily extend this orchestration to more agents: one does the code execution, one the web search, one handles file loadings...


<EditOnGithub source="https://github.com/huggingface/smolagents/blob/main/docs/source/en/examples/multiagents.md" />



# Agentic RAG

<DocNotebookDropdown
  classNames="absolute z-10 right-0 top-0"
  options={[
    {label: "Mixed", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/main/smolagents_doc/en/rag.ipynb"},
    {label: "PyTorch", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/main/smolagents_doc/en/pytorch/rag.ipynb"},
    {label: "TensorFlow", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/main/smolagents_doc/en/tensorflow/rag.ipynb"},
    {label: "Mixed", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/smolagents_doc/en/rag.ipynb"},
    {label: "PyTorch", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/smolagents_doc/en/pytorch/rag.ipynb"},
    {label: "TensorFlow", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/smolagents_doc/en/tensorflow/rag.ipynb"},
]} />

## Introduction to Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) combines the power of large language models with external knowledge retrieval to produce more accurate, factual, and contextually relevant responses. At its core, RAG is about "using an LLM to answer a user query, but basing the answer on information retrieved from a knowledge base."

### Why Use RAG?

RAG offers several significant advantages over using vanilla or fine-tuned LLMs:

1. **Factual Grounding**: Reduces hallucinations by anchoring responses in retrieved facts
2. **Domain Specialization**: Provides domain-specific knowledge without model retraining
3. **Knowledge Recency**: Allows access to information beyond the model's training cutoff
4. **Transparency**: Enables citation of sources for generated content
5. **Control**: Offers fine-grained control over what information the model can access

### Limitations of Traditional RAG

Despite its benefits, traditional RAG approaches face several challenges:

- **Single Retrieval Step**: If the initial retrieval results are poor, the final generation will suffer
- **Query-Document Mismatch**: User queries (often questions) may not match well with documents containing answers (often statements)
- **Limited Reasoning**: Simple RAG pipelines don't allow for multi-step reasoning or query refinement
- **Context Window Constraints**: Retrieved documents must fit within the model's context window

## Agentic RAG: A More Powerful Approach

We can overcome these limitations by implementing an **Agentic RAG** system - essentially an agent equipped with retrieval capabilities. This approach transforms RAG from a rigid pipeline into an interactive, reasoning-driven process.

### Key Benefits of Agentic RAG

An agent with retrieval tools can:

1. ✅ **Formulate optimized queries**: The agent can transform user questions into retrieval-friendly queries
2. ✅ **Perform multiple retrievals**: The agent can retrieve information iteratively as needed
3. ✅ **Reason over retrieved content**: The agent can analyze, synthesize, and draw conclusions from multiple sources
4. ✅ **Self-critique and refine**: The agent can evaluate retrieval results and adjust its approach

This approach naturally implements advanced RAG techniques:
- **Hypothetical Document Embedding (HyDE)**: Instead of using the user query directly, the agent formulates retrieval-optimized queries ([paper reference](https://huggingface.co/papers/2212.10496))
- **Self-Query Refinement**: The agent can analyze initial results and perform follow-up retrievals with refined queries ([technique reference](https://docs.llamaindex.ai/en/stable/examples/evaluation/RetryQuery/))

## Building an Agentic RAG System

Let's build a complete Agentic RAG system step by step. We'll create an agent that can answer questions about the Hugging Face Transformers library by retrieving information from its documentation.

You can follow along with the code snippets below, or check out the full example in the smolagents GitHub repository: [examples/rag.py](https://github.com/huggingface/smolagents/blob/main/examples/rag.py).

### Step 1: Install Required Dependencies

First, we need to install the necessary packages:

```bash
pip install smolagents pandas langchain langchain-community sentence-transformers datasets python-dotenv rank_bm25 --upgrade
```

If you plan to use Hugging Face's Inference API, you'll need to set up your API token:

```python
# Load environment variables (including HF_TOKEN)
from dotenv import load_dotenv
load_dotenv()
```

### Step 2: Prepare the Knowledge Base

We'll use a dataset containing Hugging Face documentation and prepare it for retrieval:

```python
import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

# Load the Hugging Face documentation dataset
knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")

# Filter to include only Transformers documentation
knowledge_base = knowledge_base.filter(lambda row: row["source"].startswith("huggingface/transformers"))

# Convert dataset entries to Document objects with metadata
source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
    for doc in knowledge_base
]

# Split documents into smaller chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Characters per chunk
    chunk_overlap=50,  # Overlap between chunks to maintain context
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],  # Priority order for splitting
)
docs_processed = text_splitter.split_documents(source_docs)

print(f"Knowledge base prepared with {len(docs_processed)} document chunks")
```

### Step 3: Create a Retriever Tool

Now we'll create a custom tool that our agent can use to retrieve information from the knowledge base:

```python
from smolagents import Tool

class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve the parts of transformers documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        # Initialize the retriever with our processed documents
        self.retriever = BM25Retriever.from_documents(
            docs, k=10  # Return top 10 most relevant documents
        )

    def forward(self, query: str) -> str:
        """Execute the retrieval based on the provided query."""
        assert isinstance(query, str), "Your search query must be a string"

        # Retrieve relevant documents
        docs = self.retriever.invoke(query)

        # Format the retrieved documents for readability
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )

# Initialize our retriever tool with the processed documents
retriever_tool = RetrieverTool(docs_processed)
```

> [!TIP]
> We're using BM25, a lexical retrieval method, for simplicity and speed. For production systems, you might want to use semantic search with embeddings for better retrieval quality. Check the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for high-quality embedding models.

### Step 4: Create an Advanced Retrieval Agent

Now we'll create an agent that can use our retriever tool to answer questions:

```python
from smolagents import InferenceClientModel, CodeAgent

# Initialize the agent with our retriever tool
agent = CodeAgent(
    tools=[retriever_tool],  # List of tools available to the agent
    model=InferenceClientModel(),  # Default model "Qwen/Qwen2.5-Coder-32B-Instruct"
    max_steps=4,  # Limit the number of reasoning steps
    verbosity_level=2,  # Show detailed agent reasoning
)

# To use a specific model, you can specify it like this:
# model=InferenceClientModel(model_id="meta-llama/Llama-3.3-70B-Instruct")
```

> [!TIP]
> Inference Providers give access to hundreds of models, powered by serverless inference partners. A list of supported providers can be found [here](https://huggingface.co/docs/inference-providers/index).

### Step 5: Run the Agent to Answer Questions

Let's use our agent to answer a question about Transformers:

```python
# Ask a question that requires retrieving information
question = "For a transformers model training, which is slower, the forward or the backward pass?"

# Run the agent to get an answer
agent_output = agent.run(question)

# Display the final answer
print("\nFinal answer:")
print(agent_output)
```

## Practical Applications of Agentic RAG

Agentic RAG systems can be applied to various use cases:

1. **Technical Documentation Assistance**: Help users navigate complex technical documentation
2. **Research Paper Analysis**: Extract and synthesize information from scientific papers
3. **Legal Document Review**: Find relevant precedents and clauses in legal documents
4. **Customer Support**: Answer questions based on product documentation and knowledge bases
5. **Educational Tutoring**: Provide explanations based on textbooks and learning materials

## Conclusion

Agentic RAG represents a significant advancement over traditional RAG pipelines. By combining the reasoning capabilities of LLM agents with the factual grounding of retrieval systems, we can build more powerful, flexible, and accurate information systems.

The approach we've demonstrated:
- Overcomes the limitations of single-step retrieval
- Enables more natural interactions with knowledge bases
- Provides a framework for continuous improvement through self-critique and query refinement

As you build your own Agentic RAG systems, consider experimenting with different retrieval methods, agent architectures, and knowledge sources to find the optimal configuration for your specific use case.


<EditOnGithub source="https://github.com/huggingface/smolagents/blob/main/docs/source/en/examples/rag.md" />

# Agentic RAG

<DocNotebookDropdown
  classNames="absolute z-10 right-0 top-0"
  options={[
    {label: "Mixed", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/main/smolagents_doc/en/rag.ipynb"},
    {label: "PyTorch", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/main/smolagents_doc/en/pytorch/rag.ipynb"},
    {label: "TensorFlow", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/main/smolagents_doc/en/tensorflow/rag.ipynb"},
    {label: "Mixed", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/smolagents_doc/en/rag.ipynb"},
    {label: "PyTorch", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/smolagents_doc/en/pytorch/rag.ipynb"},
    {label: "TensorFlow", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/main/smolagents_doc/en/tensorflow/rag.ipynb"},
]} />

## Introduction to Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) combines the power of large language models with external knowledge retrieval to produce more accurate, factual, and contextually relevant responses. At its core, RAG is about "using an LLM to answer a user query, but basing the answer on information retrieved from a knowledge base."

### Why Use RAG?

RAG offers several significant advantages over using vanilla or fine-tuned LLMs:

1. **Factual Grounding**: Reduces hallucinations by anchoring responses in retrieved facts
2. **Domain Specialization**: Provides domain-specific knowledge without model retraining
3. **Knowledge Recency**: Allows access to information beyond the model's training cutoff
4. **Transparency**: Enables citation of sources for generated content
5. **Control**: Offers fine-grained control over what information the model can access

### Limitations of Traditional RAG

Despite its benefits, traditional RAG approaches face several challenges:

- **Single Retrieval Step**: If the initial retrieval results are poor, the final generation will suffer
- **Query-Document Mismatch**: User queries (often questions) may not match well with documents containing answers (often statements)
- **Limited Reasoning**: Simple RAG pipelines don't allow for multi-step reasoning or query refinement
- **Context Window Constraints**: Retrieved documents must fit within the model's context window

## Agentic RAG: A More Powerful Approach

We can overcome these limitations by implementing an **Agentic RAG** system - essentially an agent equipped with retrieval capabilities. This approach transforms RAG from a rigid pipeline into an interactive, reasoning-driven process.

### Key Benefits of Agentic RAG

An agent with retrieval tools can:

1. ✅ **Formulate optimized queries**: The agent can transform user questions into retrieval-friendly queries
2. ✅ **Perform multiple retrievals**: The agent can retrieve information iteratively as needed
3. ✅ **Reason over retrieved content**: The agent can analyze, synthesize, and draw conclusions from multiple sources
4. ✅ **Self-critique and refine**: The agent can evaluate retrieval results and adjust its approach

This approach naturally implements advanced RAG techniques:
- **Hypothetical Document Embedding (HyDE)**: Instead of using the user query directly, the agent formulates retrieval-optimized queries ([paper reference](https://huggingface.co/papers/2212.10496))
- **Self-Query Refinement**: The agent can analyze initial results and perform follow-up retrievals with refined queries ([technique reference](https://docs.llamaindex.ai/en/stable/examples/evaluation/RetryQuery/))

## Building an Agentic RAG System

Let's build a complete Agentic RAG system step by step. We'll create an agent that can answer questions about the Hugging Face Transformers library by retrieving information from its documentation.

You can follow along with the code snippets below, or check out the full example in the smolagents GitHub repository: [examples/rag.py](https://github.com/huggingface/smolagents/blob/main/examples/rag.py).

### Step 1: Install Required Dependencies

First, we need to install the necessary packages:

```bash
pip install smolagents pandas langchain langchain-community sentence-transformers datasets python-dotenv rank_bm25 --upgrade
```

If you plan to use Hugging Face's Inference API, you'll need to set up your API token:

```python
# Load environment variables (including HF_TOKEN)
from dotenv import load_dotenv
load_dotenv()
```

### Step 2: Prepare the Knowledge Base

We'll use a dataset containing Hugging Face documentation and prepare it for retrieval:

```python
import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

# Load the Hugging Face documentation dataset
knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")

# Filter to include only Transformers documentation
knowledge_base = knowledge_base.filter(lambda row: row["source"].startswith("huggingface/transformers"))

# Convert dataset entries to Document objects with metadata
source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
    for doc in knowledge_base
]

# Split documents into smaller chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Characters per chunk
    chunk_overlap=50,  # Overlap between chunks to maintain context
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],  # Priority order for splitting
)
docs_processed = text_splitter.split_documents(source_docs)

print(f"Knowledge base prepared with {len(docs_processed)} document chunks")
```

### Step 3: Create a Retriever Tool

Now we'll create a custom tool that our agent can use to retrieve information from the knowledge base:

```python
from smolagents import Tool

class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve the parts of transformers documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        # Initialize the retriever with our processed documents
        self.retriever = BM25Retriever.from_documents(
            docs, k=10  # Return top 10 most relevant documents
        )

    def forward(self, query: str) -> str:
        """Execute the retrieval based on the provided query."""
        assert isinstance(query, str), "Your search query must be a string"

        # Retrieve relevant documents
        docs = self.retriever.invoke(query)

        # Format the retrieved documents for readability
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )

# Initialize our retriever tool with the processed documents
retriever_tool = RetrieverTool(docs_processed)
```

> [!TIP]
> We're using BM25, a lexical retrieval method, for simplicity and speed. For production systems, you might want to use semantic search with embeddings for better retrieval quality. Check the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for high-quality embedding models.

### Step 4: Create an Advanced Retrieval Agent

Now we'll create an agent that can use our retriever tool to answer questions:

```python
from smolagents import InferenceClientModel, CodeAgent

# Initialize the agent with our retriever tool
agent = CodeAgent(
    tools=[retriever_tool],  # List of tools available to the agent
    model=InferenceClientModel(),  # Default model "Qwen/Qwen2.5-Coder-32B-Instruct"
    max_steps=4,  # Limit the number of reasoning steps
    verbosity_level=2,  # Show detailed agent reasoning
)

# To use a specific model, you can specify it like this:
# model=InferenceClientModel(model_id="meta-llama/Llama-3.3-70B-Instruct")
```

> [!TIP]
> Inference Providers give access to hundreds of models, powered by serverless inference partners. A list of supported providers can be found [here](https://huggingface.co/docs/inference-providers/index).

### Step 5: Run the Agent to Answer Questions

Let's use our agent to answer a question about Transformers:

```python
# Ask a question that requires retrieving information
question = "For a transformers model training, which is slower, the forward or the backward pass?"

# Run the agent to get an answer
agent_output = agent.run(question)

# Display the final answer
print("\nFinal answer:")
print(agent_output)
```

## Practical Applications of Agentic RAG

Agentic RAG systems can be applied to various use cases:

1. **Technical Documentation Assistance**: Help users navigate complex technical documentation
2. **Research Paper Analysis**: Extract and synthesize information from scientific papers
3. **Legal Document Review**: Find relevant precedents and clauses in legal documents
4. **Customer Support**: Answer questions based on product documentation and knowledge bases
5. **Educational Tutoring**: Provide explanations based on textbooks and learning materials

## Conclusion

Agentic RAG represents a significant advancement over traditional RAG pipelines. By combining the reasoning capabilities of LLM agents with the factual grounding of retrieval systems, we can build more powerful, flexible, and accurate information systems.

The approach we've demonstrated:
- Overcomes the limitations of single-step retrieval
- Enables more natural interactions with knowledge bases
- Provides a framework for continuous improvement through self-critique and query refinement

As you build your own Agentic RAG systems, consider experimenting with different retrieval methods, agent architectures, and knowledge sources to find the optimal configuration for your specific use case.


<EditOnGithub source="https://github.com/huggingface/smolagents/blob/main/docs/source/en/examples/rag.md" />