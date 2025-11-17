import apiClient from './api-client';
import { IAssistant,IDeleteResult, IAssistantWithStatus } from '../utils/types';
function filterAndConvert(
    assistantsWithStatus: IAssistantWithStatus[],
    statusCondition: string
  ): IAssistant[] {
    return assistantsWithStatus
      .filter((assistant) => assistant.build_status.status === statusCondition)
      .map(({ build_status, ...rest }) => rest);
  }

interface IAssistantData {
    model: string;
    prefix_system_prompt?: string;
    suffix_system_prompt?: string;
    name?: string;
    description?: string;
    tools?: any[];
    tool_resources?: object;
    metadata?:{[key:string]:any}
    top_p?: number;
    temperature?: number;
    response_format?: string;
    instructions?: string;
}

export const createAssistant = async (data: IAssistantData): Promise<IAssistant> => {
    const assistant_data: {
        model: string;
        instructions?: string;
        name?: string;
        description?: string;
        tools?: any[];
        tool_resources?: object;
        metadata?:{[key:string]:any}
        top_p?: number;
        temperature?: number;
        response_format?: string;
    } = {
        model: data.model
    };

    if (data.prefix_system_prompt) {
        assistant_data.instructions = data.prefix_system_prompt;
    }
    if (data.suffix_system_prompt) {
        assistant_data.instructions = data.suffix_system_prompt;
    }
    if (data.name) {
        assistant_data.name = data.name;
    }
    if (data.description) {
        assistant_data.description = data.description;
    }
    if (data.tools) {
        assistant_data.tools = data.tools;
    }
    if (data.tool_resources) {
        assistant_data.tool_resources = data.tool_resources;
    }
    if (data.metadata) {
        assistant_data.metadata = data.metadata
    }
    if (typeof data.top_p !== 'undefined') {
        assistant_data.top_p = data.top_p;
    }
    if (typeof data.temperature !== 'undefined') {
        assistant_data.temperature = data.temperature;
    }
    if (data.response_format) {
        assistant_data.response_format = data.response_format;
    }
    if (data.instructions) {
        assistant_data.instructions = data.instructions;
    }
    console.log(assistant_data)
    const response = await apiClient.post<IAssistant>(
        '/assistants/',
        assistant_data
    );
    console.log("response", response)
    return response.data;
};


export const listAssistants = async (
    limit?: number,
    order?: string,
    after?: string,
    before?: string,
    run_id?: string,
): Promise<IAssistant[]> => {
    const params: {
        limit?: number,
        order?: string,
        after?: string,
        before?: string,
        run_id?: string
    } = {};

    if (typeof limit !== 'undefined') {
        params.limit = limit;
    }
    if (typeof order !== 'undefined') {
        params.order = order;
    }
    if (typeof after !== 'undefined') {
        params.after = after;
    }
    if (typeof before !== 'undefined') {
        params.before = before;
    }
    if (typeof run_id !== 'undefined') {
        params.run_id = run_id;
    }
    const response = await apiClient.get<IAssistantWithStatus[]>('/assistants/status', {
        params
    });
    let tmp = response.data
    let result = [] as IAssistant[]
    const filteredAssistants = filterAndConvert(tmp, 'completed');
    return filteredAssistants
};

export const getAssistant = async (
    assistant_id: string
): Promise<IAssistant> => {
    const response = await apiClient.get<IAssistant>(`/assistants/${assistant_id}`);
    return response.data;
}

export const deleteAssistant = async (
    assistant_id: string
): Promise<IDeleteResult> => {
    const response = await apiClient.delete<IDeleteResult>(`/assistants/${assistant_id}`);
    return response.data;
}

export const getRelatedThreadId = async (
    assistant_id: string
): Promise<string[]> => {
    const response = await apiClient.get<string[]>(`/assistants/${assistant_id}/related_thread`);
    return response.data;
}

export const listAssistantsWithStatus = async (
    limit?: number,
    order?: string,
    after?: string,
    before?: string,
    run_id?: string,
): Promise<IAssistantWithStatus[]> => {
    const params: {
        limit?: number,
        order?: string,
        after?: string,
        before?: string,
        run_id?: string
    } = {};

    if (typeof limit !== 'undefined') {
        params.limit = limit;
    }
    if (typeof order !== 'undefined') {
        params.order = order;
    }
    if (typeof after !== 'undefined') {
        params.after = after;
    }
    if (typeof before !== 'undefined') {
        params.before = before;
    }
    if (typeof run_id !== 'undefined') {
        params.run_id = run_id;
    }
    console.log(params)
    const response = await apiClient.get<IAssistantWithStatus[]>('/assistants/status', {
        params
    });

    return response.data;
};


