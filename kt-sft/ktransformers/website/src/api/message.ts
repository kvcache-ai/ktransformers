import apiClient from './api-client';
import { IMessage,IDeleteResult } from '../utils/types';

export const createMessage = async (
    thread_id: string,
    content: string,
    role?: string,
    attachments?: any[],
    metadata?:{[key:string]:any}
): Promise<IMessage> => {
    const message_data: {
        content: string;
        role?: string;
        attachments?: any[];
        metadata?:{[key:string]:any}
    } = {
        content,
    };

    if (metadata) {
        message_data.metadata = metadata;
    }
    if (role) {
        message_data.role = role;
    }
    if (attachments) {
        message_data.attachments = attachments;
    }
    const response = await apiClient.post<IMessage>(`/threads/${thread_id}/messages`, message_data);
    return response.data;
};


export const listMessages = async (
    thread_id: string,
    limit?: number,
    order?: string,
    after?: string,
    before?: string,
    run_id?: string,
): Promise<IMessage[]> => {
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

    const response = await apiClient.get<IMessage[]>(`/threads/${thread_id}/messages`, {
        params
    });

    return response.data;
};
export const deleteMessage = async(thread_id:string, message_id:string): Promise<IDeleteResult> => {
    const response = await apiClient.delete<IDeleteResult>(`/threads/${thread_id}/messages/${message_id}`);
    return response.data;
}
