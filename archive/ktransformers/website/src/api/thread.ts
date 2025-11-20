import apiClient from './api-client';
import { IThread, IMessage, IThreadAndMessageAndAssistant, IDeleteResult } from '../utils/types';
export const createThread = async (
    message?: IMessage,
    tool_resources?: object,
    metadata?: { [key: string]: any }
): Promise<IThread> => {
    const thread_data: { message?: object, metadata?: { [key: string]: any } } = {};
    if (message) {
        thread_data.message = message;
    }
    if (metadata) {
        thread_data.metadata = metadata;
    }
    const response = await apiClient.post<IThread>(
        '/threads',
        thread_data);
    return response.data;
};

export const listThreads = async (
    limit?: number,
    order?: string,
): Promise<IThreadAndMessageAndAssistant[]> => {
    const params: {
        limit?: number,
        order?: string,
    } = { limit, order };
    const response = await apiClient.get<IThreadAndMessageAndAssistant[]>('/threads', {
        params
    });

    return response.data;
};

export const deleteThread = async (
    thread_id: string
): Promise<IDeleteResult> => {
    const response = await apiClient.delete<IDeleteResult>(`/threads/${thread_id}`);
    return response.data;
}

export const getThread = async (
    thread_id: string
): Promise<IThread> => {
    const response = await apiClient.get<IThread>(`/threads/${thread_id}`);
    return response.data;
}