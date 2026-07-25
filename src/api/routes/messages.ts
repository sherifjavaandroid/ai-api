import _ from 'lodash';
import process from 'process';

import Request from '@/lib/request/Request.ts';
import Response from '@/lib/response/Response.ts';
import anthropic from '@/api/controllers/anthropic.ts';

const DEEPSEEK_AUTHORIZATION = process.env.DEEPSEEK_AUTHORIZATION;
// Fallback model when the client sends a non-Qwen model name (e.g. Claude Code's
// default "claude-*" small/fast model). Override with DEEPSEEK_MESSAGES_MODEL.
const DEFAULT_MODEL = process.env.DEEPSEEK_MESSAGES_MODEL || 'deepseek-chat';

export default {

    prefix: '/v1',

    post: {

        '/messages': async (request: Request) => {
            request.validate('body.messages', _.isArray);

            // Claude Code sends the token via `x-api-key` or `Authorization: Bearer`.
            let auth = request.headers['authorization'];
            if (!auth && request.headers['x-api-key'])
                auth = 'Bearer ' + request.headers['x-api-key'];
            if (DEEPSEEK_AUTHORIZATION) auth = 'Bearer ' + DEEPSEEK_AUTHORIZATION;

            const tokens = anthropic.tokenSplit(auth || '');
            const token = _.sample(tokens);

            let { model, stream } = request.body;
            model = (model || '').toLowerCase();
            // Claude Code sends "claude-*" model names; map those to the proxy's model.
            if (!model || model.startsWith('claude')) model = DEFAULT_MODEL;

            if (stream) {
                const s = await anthropic.createMessagesStream(model, request.body, token);
                return new Response(s, { type: 'text/event-stream' });
            }
            return await anthropic.createMessages(model, request.body, token);
        }

    }

}
