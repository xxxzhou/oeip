using OeipCommon.OeipAttribute;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace OeipControl.Controls
{
    public class BaseControl<T> : UserControl, IOeipComponent, IOeipComponent<T>
    {
        public Action<T, BaseControl<T>> onValueChangeAction;
        protected Label label;
        private string displayName = string.Empty;
        private ControlAttribute controlAttribute = null;

        public BaseControl()
        {
            InitializeComponent();
        }

        public string DisplayName
        {
            get
            {
                return displayName;
            }
            set
            {
                displayName = value;
                label.Text = displayName;
            }
        }

        public ControlAttribute ControlAttribute
        {
            get
            {
                return controlAttribute;
            }
            set
            {
                controlAttribute = value;
                OnSetAttribute();
            }
        }

        public U Attribute<U>() where U : ControlAttribute
        {
            if (ControlAttribute == null)
                return null;
            U t = ControlAttribute as U;
            return t;
        }

        public virtual void OnSetAttribute()
        {

        }

        //用值去更新UI控件
        public virtual void OnSetValue(T value)
        {
        }

        public void SetValueChangeAction(Action<T, IOeipComponent<T>> onAction)
        {
            onValueChangeAction = onAction;
        }

        public void OnValueChange(T value)
        {
            onValueChangeAction?.Invoke(value, this);
        }

        public virtual void UpdateControl(object obj)
        {
            if (ControlAttribute == null)
                return;
            T value = ControlAttribute.GetValue<T>(ref obj);
            OnSetValue(value);
        }

        private void InitializeComponent()
        {
            this.SuspendLayout();
            // 
            // BaseControl
            // 
            this.Name = "BaseControl";
            this.Size = new System.Drawing.Size(318, 24);
            this.ResumeLayout(false);
        }
    }
}
