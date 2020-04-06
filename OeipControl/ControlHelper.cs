using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace OeipControl
{
    public static class ControlHelper
    {
        public static void TryBeginInvoke(this Control control, Action action)
        {
            try
            {
                control.BeginInvoke(action);
            }
            catch (Exception ex)
            {

            }
        }
    }
}
