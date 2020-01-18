using Microsoft.Owin;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OeipLiveServer
{
    using AppFunc = Func<IDictionary<string, object>, Task>;

    public class OeipLiveMiddleware
    {
        private readonly AppFunc _next;
        public OeipLiveMiddleware(AppFunc next)
        {
            _next = next;
        }

        public static T Get<T>(IDictionary<string, object> env, string key)
        {
            object value;
            return env.TryGetValue(key, out value) ? (T)value : default(T);
        }

        public Task Invoke(IDictionary<string, object> env)
        {
            IOwinContext context = new OwinContext(env);
            context.Response.Headers["Content-Type"] = "text/html";// new string[] { "text/html" };
            string path = Get<string>(env, "owin.RequestPath");
            if (path == "/")
            {
                Stream output = Get<Stream>(env, "owin.ResponseBody");
                using (var writer = new StreamWriter(output))
                {
                    writer.Write("<ul>");
                    writer.Write("<li><a href='/Welcome'>/Welcome</a> Welcome Page</li>");
                    writer.Write("</ul>");
                }
                return Task.FromResult<object>(null);
            }
            else
            {
                return _next(env);
            }
        }
    }
}
