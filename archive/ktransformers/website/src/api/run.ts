import apiClient from './api-client';
import { IRun } from '../utils/types';
import {baseURL} from '@/conf/config';
interface IRunData {
    assistant_id: string;
    model?: string;
    instructions?: string;
    additional_instructions?: string;
    additional_messages?: any[];
    tools?: any[];
    metadata?: { [key: string]: any }
    temperature?: number;
    top_p?: number;
    stream?: boolean;
    max_prompt_tokens?: number;
    max_completion_tokens?: number;
    truncation_strategy?: object;
    tool_choice?: string;
    response_format?: string | object;
}


export async function* createRun(
    data: IRunData,
    thread_id: string
): AsyncGenerator<string> {
    const run_data = {
        ...data, 
        assistant_id: data.assistant_id, 
    };

    const response = await fetch(`${baseURL}/threads/${thread_id}/runs`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(run_data),
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    if (!response.body) {
        throw new Error('Response body is missing');
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) return;
            buffer += decoder.decode(value, { stream: true });

            let eventIndex = buffer.indexOf("\n\n");
            while (eventIndex !== -1) {
                const event = buffer.slice(0, eventIndex);
                buffer = buffer.slice(eventIndex + 2);
                if (event.startsWith("event: thread.run.created")) {
                    const dataIndex = event.indexOf("data: ");
                    if (dataIndex !== -1) {
                        const datads = event.slice(39, 75)
                        yield datads;
                    }
                } else if (event.startsWith("event: thread.message.delta")) {
                    const dataIndex = event.indexOf("data: ");
                    if (dataIndex !== -1) {
                        const data = JSON.parse(event.slice(dataIndex + 6));
                        yield data.delta.content[0].text.value || '';
                    }
                } else if (event.startsWith("event: done")) {
                    return;
                }

                eventIndex = buffer.indexOf("\n\n");
            }
        }
    } catch (e) {

        console.error('An error occurred while reading the response stream:', e);
        // throw e; 
        return e
    }
}
// 定义取消运行的函数
export async function cancelRun(threadId: string, runId: string){
    const run_data = {
        thread_id:threadId,
        run_id:runId,
    };
    try {
        const response = await fetch(`${baseURL}/threads/${threadId}/runs/${runId}/cancel`, {
            method: 'POST',
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return response;
    } catch (error) {
        console.error('An error occurred while cancelling the run:', error);
        throw error;
    }
}