from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from tiktoken import encoding_for_model
from typing import List

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class TeacherAssistant:
    supabase: Client
    openai_client: AsyncOpenAI

system_prompt = """
You are an AI assistant designed to help educators manage and update their course content effectively.
Your role is to support educators in maintaining clear, consistent, and up-to-date course materials with minimal manual effort.

The course content is organized using knowledge graphs—visual maps where nodes represent key knowledge units connected 
by relationships that show how concepts relate to each other. These knowledge points are aligned with cognitive levels (cog_levels if you would), 
which specify what students should learn or be able to do by the end of the course or learning unit.

There are also syllabi voor the courses. You can use these for more general questions about the course, or when they ask for it specifically.


"""
graph_ai_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=TeacherAssistant,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error


@graph_ai_expert.tool
async def compare_2_KGs(ctx: RunContext[TeacherAssistant], user_query: str, match_count: int = 5) -> str:
    """
    Retrieve relevant information about 2 given courses and compare 
    their Knowledge Graphs(KGs) to give a relevant answer to the question of the educator.
    """
    try:
        query_embedding1 = await get_embedding(user_query[0], ctx.deps.openai_client)
        query_embedding2 = await get_embedding(user_query[1], ctx.deps.openai_client)
        result1 = ctx.deps.supabase.rpc(
            'match_graph_nodes',
            {
                'query_embedding': query_embedding1,
                'match_count': match_count,
                'filter': {}
            }
        ).execute()

        if not result1.data:
            return "No relevant nodes found."
        formatted_nodes = []
        for node in result1.data:
            node_text = f"""
            **node_id**: {node['node_id']}  
            **URL**: {node['url']}
            **course**: {node['course']}    
            **Label**: {node.get('label', 'N/A')}  
            **Cognitive Level**: {node.get('cog_level', 'N/A')}  
            **Node Type**: {node.get('node_type', 'N/A')} 
            **Node Description**: {node['node_description']}  
            **Metadata**: {node.get('node_metadata', {})}
            """         
            formatted_nodes.append(node_text)
        safe_chunks1 = trim_to_token_limit(formatted_nodes, llm, 2500)
        result2 = ctx.deps.supabase.rpc(
            'match_graph_nodes',
            {
                'query_embedding': query_embedding2,
                'match_count': match_count,
                'filter': {}
            }
        ).execute()

        if not result2.data:
            return "No relevant nodes found."
        formatted_nodes = []
        for node in result2.data:
            node_text = f"""
            **node_id**: {node['node_id']}  
            **URL**: {node['url']}
            **course**: {node['course']}    
            **Label**: {node.get('label', 'N/A')}  
            **Cognitive Level**: {node.get('cog_level', 'N/A')}  
            **Node Type**: {node.get('node_type', 'N/A')} 
            **Node Description**: {node['node_description']}  
            **Metadata**: {node.get('node_metadata', {})}
            """         
            formatted_nodes.append(node_text)
        safe_chunks2 = trim_to_token_limit(formatted_nodes, llm, 2500)
        comparison_prompt = f"""
        Compare the following two course KG node sets. Identify any shared topics, overlapping concepts, or similarities in skills or learning objectives.
        Course: {user_query[0]}
        {safe_chunks1}

        ---
        Course: {user_query[1]}
        {safe_chunks2}

        ---

        Provide a thoughtful comparison and highlight any overlaps or significant differences.
        """

        return comparison_prompt
    except Exception as e:
        return f"Error comparing information: {str(e)}"

@graph_ai_expert.tool
async def retrieve_syllabi_information(ctx: RunContext[TeacherAssistant], user_query: str, match_count: int = 5) -> str:
    """
    Retrieve relevant information about the syllabus of the given course. 
    Use this information to generate a relevant answer.
    """
    try:
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        result = ctx.deps.supabase.rpc(
            'retrieve_syllabi',
            {
                'query_embedding': query_embedding,
                'match_count': match_count,
                'filter': {}
            }
        ).execute()

        if not result.data:
            return "No relevant data found."
        formatted_data = []
        for node in result.data:
            node_text = f"""
            **URL**: {node['url']}
            **course**: {node['title']} 
            **Content**: {node.get('content', 'N/A')}   
            """         
            formatted_data.append(node_text)
        safe_chunks = trim_to_token_limit(formatted_data, llm, 2500)
        prompt = f"""
        Use the syllabus information down to retrieve relevant information.
        Course: {user_query}
        {safe_chunks}
        Provide a relevant answer  corresponding to the question of the educator.
        """
        return prompt

    except Exception as e:
        print(f"Error retrieving nodes: {e}")
        return f"Error retrieving nodes: {str(e)}"


@graph_ai_expert.tool
async def compare_syllabi(ctx: RunContext[TeacherAssistant], user_query: str, match_count: int = 5) -> str:
    """
    Retrieve relevant information about the syllabi of the two given courses. 
    Use this information to generate a relevant answer and compare them.
    """
    try:
        query_embedding1 = await get_embedding(user_query[0], ctx.deps.openai_client)
        query_embedding2 = await get_embedding(user_query[1], ctx.deps.openai_client)
        result = ctx.deps.supabase.rpc(
            'retrieve_syllabi',
            {
                'query_embedding': query_embedding1,
                'match_count': match_count,
                'filter': {}
            }
        ).execute()

        if not result.data:
            return "No relevant data found."
        formatted_data = []
        for node in result.data:
            node_text = f"""
            **URL**: {node['url']}
            **course**: {node['title']} 
            **Content**: {node.get('content', 'N/A')}   
            """         
            formatted_data.append(node_text)
        safe_chunks1 = trim_to_token_limit(formatted_data, llm, 2500)

        result = ctx.deps.supabase.rpc(
            'retrieve_syllabi',
            {
                'query_embedding': query_embedding2,
                'match_count': match_count,
                'filter': {}
            }
        ).execute()

        if not result.data:
            return "No relevant data found."
        formatted_data = []
        for node in result.data:
            node_text = f"""
            **URL**: {node['url']}
            **course**: {node['title']} 
            **Content**: {node.get('content', 'N/A')}   
            """         
            formatted_data.append(node_text)
        safe_chunks2 = trim_to_token_limit(formatted_data, llm, 2500)

        comparison_prompt = f"""
        Compare the following two course syllabi sets. Identify any shared topics, overlapping concepts, or similarities in skills or learning objectives.
        Course: {user_query[0]}
        {safe_chunks1}

        ---
        Course: {user_query[1]}
        {safe_chunks2}

        ---

        Provide a thoughtful comparison and highlight any overlaps or significant differences.
        """

        return comparison_prompt

    except Exception as e:
        print(f"Error retrieving nodes: {e}")
        return f"Error retrieving nodes: {str(e)}"
    
@graph_ai_expert.tool
async def compare_syllabi_KG(ctx: RunContext[TeacherAssistant], user_query: str, match_count: int = 5) -> str:
    """
    Retrieve relevant information about the syllabi and the KG of the given course. 
    Use this information to generate a relevant answer and compare them.
    """
    try:
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        result1 = ctx.deps.supabase.rpc(
            'retrieve_syllabi',
            {
                'query_embedding': query_embedding,
                'match_count': match_count,
                'filter': {}
            }
        ).execute()

        if not result1.data:
            return "No relevant data found."
        formatted_data = []
        for node in result1.data:
            node_text = f"""
            **URL**: {node['url']}
            **course**: {node['title']} 
            **Content**: {node.get('content', 'N/A')}   
            """         
            formatted_data.append(node_text)
        safe_chunks1 = trim_to_token_limit(formatted_data, llm, 2500)

        result2 = ctx.deps.supabase.rpc(
            'match_graph_nodes',
            {
                'query_embedding': query_embedding,
                'match_count': match_count,
                'filter': {}
            }
        ).execute()

        if not result2.data:
            return "No relevant nodes found."
        formatted_nodes = []
        for node in result2.data:
            node_text = f"""
            **node_id**: {node['node_id']}  
            **URL**: {node['url']}
            **course**: {node['course']}    
            **Label**: {node.get('label', 'N/A')}  
            **Cognitive Level**: {node.get('cog_level', 'N/A')}  
            **Node Type**: {node.get('node_type', 'N/A')} 
            **Node Description**: {node['node_description']}  
            **Metadata**: {node.get('node_metadata', {})}
            """         
            formatted_nodes.append(node_text)
        safe_chunks2 = trim_to_token_limit(formatted_nodes, llm, 2500)

        comparison_prompt = f"""
        Compare the syllabi and KG of the given course. Identify any shared topics, overlapping concepts, or differences in skills or learning objectives.
        Course: {user_query} Syllabi:
        {safe_chunks1}

        ---
        Course: {user_query} KG:
        {safe_chunks2}

        ---

        Provide a thoughtful comparison and highlight any overlaps or significant differences.
        """

        return comparison_prompt

    except Exception as e:
        print(f"Error retrieving nodes: {e}")
        return f"Error retrieving nodes: {str(e)}"

@graph_ai_expert.tool
async def retrieve_relevant_nodes(ctx: RunContext[TeacherAssistant], user_query: str, match_count: int = 5) -> str:
    """
    Retrieve relevant graph nodes corresponding to the given course. 
    Use this information to give a relevant answer to the question of the educator.
    """
    try:
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        result = ctx.deps.supabase.rpc(
            'match_graph_nodes',
            {
                'query_embedding': query_embedding,
                'match_count': match_count,
                'filter': {}
            }
        ).execute()

        if not result.data:
            return "No relevant nodes found."
        formatted_nodes = []
        for node in result.data:
            node_text = f"""
            **node_id**: {node['node_id']}  
            **URL**: {node['url']}
            **course**: {node['course']}    
            **Label**: {node.get('label', 'N/A')}  
            **Cognitive Level**: {node.get('cog_level', 'N/A')}  
            **Node Type**: {node.get('node_type', 'N/A')} 
            **Node Description**: {node['node_description']}  
            **Metadata**: {node.get('node_metadata', {})}
            """         
            formatted_nodes.append(node_text)
        safe_chunks = trim_to_token_limit(formatted_nodes, llm, 2500)
        return "\n\n---\n\n".join(safe_chunks)

    except Exception as e:
        print(f"Error retrieving nodes: {e}")
        return f"Error retrieving nodes: {str(e)}"


@graph_ai_expert.tool
async def retrieve_relevant_edges(ctx: RunContext[TeacherAssistant], user_query: str, match_count: int = 5) -> str:
    """
    Retrieve relevant graph edges based on the query using vector similarity search.
    """
    try:
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)

        result = ctx.deps.supabase.rpc(
            'match_graph_edges',
            {
                'query_embedding': query_embedding,
                'match_count': match_count,
                'filter': {},  # You can use metadata filters here too
            }
        ).execute()

        if not result.data:
            return "No relevant edges found."

        formatted_edges = []
        for edge in result.data:
            edge_text = f"""
            **Edge_id**: {edge['edge_id']}  
            **Course**: {edge['course']} 
            **URL**: {edge['url']}   
            **Source Node ID**: {edge['source_id']}  
            **Source Node label**: {edge['source_label']}
            **Target Node ID**: {edge['target_id']}  
            **Target Node label**: {edge['target_label']} 
            **Metadata**: {edge.get('edge_metadata', {})}
            """
            formatted_edges.append(edge_text)
        safe_chunks = trim_to_token_limit(formatted_edges, llm, 2500)
        return "\n\n---\n\n".join(safe_chunks)

    except Exception as e:
        print(f"Error retrieving edges: {e}")
        return f"Error retrieving edges: {str(e)}"


def trim_to_token_limit(texts: List[str], model_name: str, max_tokens: int) -> List[str]:
    enc = encoding_for_model(model_name)
    result = []
    total_tokens = 0
    for text in texts:
        tokens = len(enc.encode(text))
        if total_tokens + tokens > max_tokens:
            break
        result.append(text)
        total_tokens += tokens
    return result
