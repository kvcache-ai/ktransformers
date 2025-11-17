declare module '*.js' {
    const config: {
      apiUrl: string;
      port:number;
    };
    export { config };
  }