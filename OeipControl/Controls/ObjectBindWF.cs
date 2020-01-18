using OeipCommon.OeipAttribute;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace OeipControl.Controls
{
    public class ObjectBindWF<T> : ObjectBind<T>
    {
        public override IOeipComponent CreateComponent(ControlAttribute attribute)
        {
            if (attribute is SliderInputAttribute)
            {
                return new SliderInputControl();
            }
            else if (attribute is InputAttribute)
            {
                return new InputControl();
            }
            else if (attribute is ToggleAttribute)
            {
                return new ToggleControl();
            }
            else if (attribute is DropdownAttribute)
            {
                return new DropdownControl();
            }
            return null;
        }

        public override bool OnAddPanel<P>(IOeipComponent component, P panel)
        {
            if (panel == null)
                return false;
            if (panel is FlowLayoutPanel)
            {
                UserControl userControl = component as UserControl;
                if (userControl == null)
                    return false;
                FlowLayoutPanel flowLayoutPanel = panel as FlowLayoutPanel;
                flowLayoutPanel.Controls.Add(userControl);
                return true;
            }
            return false;
        }

        public void Bind(T t, FlowLayoutPanel panel, Action<T> action = null)
        {
            Bind<FlowLayoutPanel>(t, panel, action);
        }

        public override void OnBind()
        {
            foreach (var comp in components)
            {
                if (comp is DropdownControl)
                {
                    DropdownControl dc = comp as DropdownControl;
                    DropdownAttribute da = comp.ControlAttribute as DropdownAttribute;
                    if (!string.IsNullOrEmpty(da.Parent))
                    {
                        var parent = GetComponent(da.Parent) as DropdownControl;
                        if (parent != null)
                        {
                            dc.parent = parent;
                        }
                    }
                }
            }
        }
    }
}
