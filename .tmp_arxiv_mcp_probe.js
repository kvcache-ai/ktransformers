const { spawn } = require('child_process');

const env = {
  ...process.env,
  WORK_DIR: '/Users/qr/Documents/papers/arxiv',
};

const child = spawn('node', ['/Users/qr/.codex/mcp-arxiv/run-arxiv-mcp-wrapper.mjs'], {
  env,
  stdio: ['pipe', 'pipe', 'pipe'],
});

let nextId = 1;
let buf = Buffer.alloc(0);
const pending = new Map();

function send(msg) {
  const s = JSON.stringify(msg);
  const payload = `Content-Length: ${Buffer.byteLength(s)}\r\n\r\n${s}`;
  child.stdin.write(payload);
}

function req(method, params) {
  const id = nextId++;
  send({ jsonrpc: '2.0', id, method, params });
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    setTimeout(() => {
      if (pending.has(id)) {
        pending.delete(id);
        reject(new Error(`timeout ${method}`));
      }
    }, 15000);
  });
}

function parse() {
  while (true) {
    const s = buf.toString('utf8');
    const i = s.indexOf('\r\n\r\n');
    if (i === -1) return;
    const header = s.slice(0, i);
    const m = header.match(/Content-Length:\s*(\d+)/i);
    if (!m) {
      const nl = s.indexOf('\n');
      if (nl === -1) return;
      console.error('NOISE:', s.slice(0, nl).trim());
      buf = Buffer.from(s.slice(nl + 1), 'utf8');
      continue;
    }
    const len = Number(m[1]);
    const total = i + 4 + len;
    if (buf.length < total) return;
    const body = buf.slice(i + 4, total).toString('utf8');
    buf = buf.slice(total);
    let msg;
    try {
      msg = JSON.parse(body);
    } catch (e) {
      console.error('JSONERR:', body);
      continue;
    }
    if (msg.id && pending.has(msg.id)) {
      const p = pending.get(msg.id);
      pending.delete(msg.id);
      if (msg.error) {
        p.reject(new Error(JSON.stringify(msg.error)));
      } else {
        p.resolve(msg.result);
      }
    } else {
      console.log('UNSOLICITED', JSON.stringify(msg));
    }
  }
}

child.stdout.on('data', (c) => {
  buf = Buffer.concat([buf, c]);
  parse();
});

child.stderr.on('data', (c) => process.stderr.write(`STDERR:${c.toString()}`));

child.on('exit', (code) => {
  console.error('EXIT', code);
});

(async () => {
  try {
    const init = await req('initialize', {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: { name: 'probe', version: '0.1' },
    });
    console.log('INIT_OK');
    console.log(JSON.stringify(init, null, 2));

    send({ jsonrpc: '2.0', method: 'notifications/initialized', params: {} });

    const tools = await req('tools/list', {});
    console.log('TOOLS_OK');
    console.log(JSON.stringify(tools, null, 2));
  } catch (e) {
    console.error('ERR', e.message);
    process.exitCode = 1;
  } finally {
    child.kill('SIGTERM');
    setTimeout(() => child.kill('SIGKILL'), 1000);
  }
})();
