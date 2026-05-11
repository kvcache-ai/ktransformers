import axios, { AxiosInstance } from 'axios';
import {baseURL} from '@/conf/config';
const apiClient: AxiosInstance = axios.create({
    baseURL: baseURL,
    // baseURL: '/api',
    headers: {
        'Content-Type': 'application/json',
    },
    withCredentials: true,
});
export default apiClient;
