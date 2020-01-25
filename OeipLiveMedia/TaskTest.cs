using OeipCommon;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

//async void 普通调用

namespace OeipLiveMedia
{
    public class TaskTest
    {
        public void Test()
        {
            LogHelper.LogMessage("0", OeipLogLevel.OEIP_WARN);
            Test7();
            Thread.Sleep(1000);
            LogHelper.LogMessage("1", OeipLogLevel.OEIP_WARN);
            Thread.Sleep(1000);
            LogHelper.LogMessage("2", OeipLogLevel.OEIP_WARN);
            Thread.Sleep(1000);
            LogHelper.LogMessage("3", OeipLogLevel.OEIP_WARN);
            Thread.Sleep(1000);
            LogHelper.LogMessage("4", OeipLogLevel.OEIP_WARN);
            Thread.Sleep(1000);
            LogHelper.LogMessage("5", OeipLogLevel.OEIP_WARN);
        }

        public void Test1()
        {
            Task.Run(() =>
            {
                LogHelper.LogMessage("x111");
                Thread.Sleep(1000);
                LogHelper.LogMessage("1111");
                Thread.Sleep(1000);
                LogHelper.LogMessage("2111");
            });
            LogHelper.LogMessage("0111");
        }

        //Test2线程内同步执行,而Test2外部线程异步
        public async void Test2()
        {
            await Task.Run(() =>
            {
                LogHelper.LogMessage("x111");
                Thread.Sleep(1000);
                LogHelper.LogMessage("1111");
                Thread.Sleep(1000);
                LogHelper.LogMessage("2111");
            });
            await Task.Run(() =>
            {
                LogHelper.LogMessage("px111");
                Thread.Sleep(1000);
                LogHelper.LogMessage("p1111");
                Thread.Sleep(1000);
                LogHelper.LogMessage("p2111");
            });
            LogHelper.LogMessage("0111");
        }

        public void Test3()
        {
            Task.Run(() =>
            {
                LogHelper.LogMessage("x111");
                Thread.Sleep(1000);
                LogHelper.LogMessage("1111");
                Thread.Sleep(1000);
                LogHelper.LogMessage("2111");
            }).Wait();
            Task.Run(() =>
            {
                LogHelper.LogMessage("px111");
                Thread.Sleep(1000);
                LogHelper.LogMessage("p1111");
                Thread.Sleep(1000);
                LogHelper.LogMessage("p2111");
            }).Wait();
            LogHelper.LogMessage("0111");
        }

        private async Task Test41()
        {
            await Task.Run(() =>
            {
                Thread.Sleep(1000);
                LogHelper.LogMessage("x111");
            });
            LogHelper.LogMessage("0111");
            Thread.Sleep(1000);
            LogHelper.LogMessage("3111");
            await Task.Run(() =>
            {
                Thread.Sleep(1000);
                LogHelper.LogMessage("x222");
                Thread.Sleep(1000);
            });
            LogHelper.LogMessage("4111");
        }

        public async void Test4()
        {
            await Test41();
        }

        private Task<int> Test51()
        {
            Thread.Sleep(3000);
            return Task.FromResult(3);
        }

        public async void Test5()
        {
            LogHelper.LogMessage("1111");
            //有点不一样，测试几次现象暂时是同步内外线程(如果后面有await void/task就会内外线程又异步)
            int x = await Test51();
            await Task.Run(() =>
            {
                LogHelper.LogMessage("0000");
            });
            LogHelper.LogMessage("xx:" + x);
            //因为前面有await void/task,所以在这又只同步本线程，与外部线程异步?
            x = await Test51();
            LogHelper.LogMessage("1111");
            LogHelper.LogMessage("xx:" + x);
        }

        ////现在的限制是，异步内外线程并返回结果 二者不可兼得，本质还是开个线程，待线程结果后执行回调
        public async void Test6()
        {
            LogHelper.LogMessage("1111");
            int x = 0;
            //这样可以异步内外线程并返回值
            await Task.Run(() =>
            {
                x = Test51().Result;
            });
            LogHelper.LogMessage("xx:" + x);
            x = await Test51();
            LogHelper.LogMessage("1111");
            LogHelper.LogMessage("xx:" + x);
        }

        private Task Test71(int i)
        {
            return Task.Run(() =>
            {
                LogHelper.LogMessage($"{i}-x111");
                Thread.Sleep(5000);
                LogHelper.LogMessage($"{i}-x222");
            });
        }

        private Task Test72(int i)
        {
            LogHelper.LogMessage($"{i}-a111");
            Thread.Sleep(1100);
            return Task.Run(() =>
            {
                Thread.Sleep(2100);
                LogHelper.LogMessage($"{i}-a222");
            }).ContinueWith((Task task) =>
            {
                Thread.Sleep(1100);
                LogHelper.LogMessage($"{i}-a333");
            });
        }

        public void Test7()
        {
            Test72(0);
            LogHelper.LogMessage("y111");
            //这个wait只针对返回的task内部运行时间
            bool result = Test72(1).Wait(3000);
            LogHelper.LogMessage($"{result}-y222");
            //在等待时间内返回结果就是true
            result = Test72(2).Wait(6000);
            LogHelper.LogMessage($"{result}-y333");
            Test71(2);
            LogHelper.LogMessage("y444");
        }
    }
}
