import _ from 'lodash';
import process from 'process';

import Request from '@/lib/request/Request.ts';
import Response from '@/lib/response/Response.ts';
import agent from '@/api/controllers/agent.ts';

const DEEPSEEK_AUTHORIZATION = process.env.DEEPSEEK_AUTHORIZATION;

export default {

    prefix: '/v1',

    post: {

        '/responses': async (request: Request) => {
            request.validate('headers.authorization', _.isString);
            if (DEEPSEEK_AUTHORIZATION)
                request.headers.authorization = 'Bearer ' + DEEPSEEK_AUTHORIZATION;
            const tokens = agent.tokenSplit(request.headers.authorization);
            const token = _.sample(tokens);
            let { model, stream } = request.body;
            model = (model || 'deepseek-chat').toLowerCase();
            if (stream) {
                const responseStream = await agent.createResponsesStream(model, request.body, token);
                return new Response(responseStream, { type: 'text/event-stream' });
            }
            return await agent.createResponses(model, request.body, token);
        }

    }

}
