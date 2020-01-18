using System;
using System.Threading.Tasks;
using Microsoft.AspNet.SignalR;
using Microsoft.Owin;
using Microsoft.Owin.Cors;
using Microsoft.Owin.Diagnostics;
using Owin;

[assembly: OwinStartup(typeof(OeipLiveServer.Startup))]
namespace OeipLiveServer
{
    public class Startup
    {
        public void Configuration(IAppBuilder app)
        {
            app.Use<OeipLiveMiddleware>();
            app.UseErrorPage(new ErrorPageOptions { SourceCodeLineCount = 20 });
            app.UseWelcomePage("/Welcome");
            var config = new HubConfiguration
            {
                EnableJSONP = true,
                EnableDetailedErrors = true,
                EnableJavaScriptProxies = true
            };
            app.UseCors(CorsOptions.AllowAll);
            app.MapSignalR();
        }
    }
}
