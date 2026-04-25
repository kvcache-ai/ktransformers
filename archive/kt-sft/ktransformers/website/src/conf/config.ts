declare global {
    interface Window {
      configWeb: {
        apiUrl: string;
        port: string;
       };
     }
  }

export const baseURL = window.configWeb.apiUrl;
export const basePort = window.configWeb.port;
