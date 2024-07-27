export interface IAssistant {
  id: string;
  object: string;
  created_at: number;
  name?: string;
  description?: string;
  model: string;
  instructions?: string;
  tools: any[];
  tool_resources?: object;
  metadata?:{[key:string]:any}
  top_p?: number;
  temperature?: number;
  response_format: string | object;
}

export interface IAssistantWithStatus {
  build_status:{status:string}
  id: string;
  object: string;
  created_at: number;
  name?: string;
  description?: string;
  model: string;
  instructions?: string;
  tools: any[];
  tool_resources?: object;
  metadata?:{[key:string]:any}
  top_p?: number;
  temperature?: number;
  response_format: string | object;
}

export interface IMessage {
  id: string;
  object: string;
  created_at: number;
  thread_id: string;
  status: string;
  incomplete_details?: object;
  completed_at?: number;
  incomplete_at?: number;
  role: string;
  content: any[];
  assistant_id?: string;
  run_id?: string;
  attachments?: any[];
  metadata:{[key:string]:any}
}

export interface IThread {
  id: string;
  object: string;
  created_at: number;
  tool_resources?: object;
  metadata?:{[key:string]:any}
}

export interface IRun {
  id: string;
  object: string;
  created_at: number;
  thread_id: string,
  assistant_id: string,
  status: string,
  required_action?: object,
  last_error?: object,
  expires_at?: number,
  started_at?: number,
  cancelled_at?: number,
  failed_at?: number,
  completed_at?: number,
  incomplete_details?: object,
  model: string,
  instructions: string,
  tools: any[],
  metadata: Map<string, string>,
  usage?: object,
  temperature?: number,
  top_p?: number,
  max_prompt_tokens?: number,
  max_completion_tokens?: number,
  truncation_strategy: object,
  tool_choice: string | object,
  response_format: string | object,
}

export interface IFile {
  id: string,
  bytes: number,
  created_at: number,
  filename: string,
  object: string,
  purpose: string,
}

export interface IMessageData {
  role: string;
  content: any[];
  created_at?: number;
  assistant_id?: string,
}

export interface IThreadAndMessageAndAssistant {

  thread: IThread;
  first_message: IMessage;
  assistant: IAssistantWithStatus
}
export interface IDeleteResult {
  id: string;
  object: string;
  deleted: boolean;
}
export interface IBuildData {
  parsed_file_count:number;
  total_file_count:number;
  prefilling_current:number;
  prefilling_total:number;
  build_completed_time:number;
  build_started_time:number;
  storage_total:number;
  storage_usage:number;
  status:string
}