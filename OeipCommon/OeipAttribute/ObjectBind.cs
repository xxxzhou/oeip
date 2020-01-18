using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OeipCommon.OeipAttribute
{
    public abstract class ObjectBind<T>
    {
        private T obj;

        protected List<ControlAttribute> attributes = null;
        protected List<IOeipComponent> components = null;
        /// <summary>
        /// 如果T是结构,每次更新后需要手动拿到Obj才能拿到最新值,可以考虑注册一个回调，自动更新
        /// </summary>
        protected Action<T> onAction = null;
        /// <summary>
        /// 当关联控件属性值发生更新回调
        /// </summary>
        public event Action<ObjectBind<T>, string> OnChangeEvent;

        public T Obj => obj;

        public abstract IOeipComponent CreateComponent(ControlAttribute attribute);

        public abstract bool OnAddPanel<P>(IOeipComponent component, P panel);

        /// <summary>
        /// 绑定一个结构/类到一个Panel上面
        /// </summary>
        /// <typeparam name="F"></typeparam>
        /// <param name="t"></param>
        /// <param name="panel"></param>
        /// <param name="onTemplate"></param>
        /// <param name="onAddPanel"></param>
        /// <param name="action"></param>
        public void Bind<P>(T t, P panel, Action<T> action = null)
        {
            obj = t;
            attributes = OeipAttributeHelper.GetAttributes(t);
            onAction = action;
            components = new List<IOeipComponent>();
            foreach (var attribute in attributes)
            {
                IOeipComponent component = CreateComponent(attribute);
                if (component == null)
                    continue;
                bool bAdd = OnAddPanel(component, panel);
                if (!bAdd)
                    continue;
                if (component is IOeipComponent<int>)
                    SetComponent<IOeipComponent<int>, int>(component as IOeipComponent<int>, attribute);
                if (component is IOeipComponent<float>)
                    SetComponent<IOeipComponent<float>, float>(component as IOeipComponent<float>, attribute);
                if (component is IOeipComponent<string>)
                    SetComponent<IOeipComponent<string>, string>(component as IOeipComponent<string>, attribute);
                if (component is IOeipComponent<bool>)
                    SetComponent<IOeipComponent<bool>, bool>(component as IOeipComponent<bool>, attribute);
                components.Add(component);
            }
            //更新所有控件值的显示
            Update();
            //让子类在这里也可以做些事
            OnBind();
        }

        public void SetComponent<A, B>(A component, ControlAttribute attribute) where A : IOeipComponent<B>
        {
            component.ControlAttribute = attribute;
            component.DisplayName = attribute.DisplayName;
            component.SetValueChangeAction(OnValueChange);
        }

        private void OnValueChange<A>(A value, IOeipComponent<A> component)
        {
            component.ControlAttribute.SetValue(ref obj, value);
            onAction?.Invoke(obj);
            OnChangeEvent?.Invoke(this, component.ControlAttribute.Member.Name);
        }

        /// <summary>
        /// 更新所有UI的值
        /// </summary>
        public void Update()
        {
            if (components == null)
                return;
            foreach (var comp in components)
            {
                comp.UpdateControl(obj);
            }
        }

        public void Update(T t)
        {
            obj = t;
            Update();
        }

        public IOeipComponent GetComponent(string memberName)
        {
            foreach (var comp in components)
            {
                if (comp.ControlAttribute.Member.Name == memberName)
                {
                    return comp;
                }
            }
            return null;
        }

        public virtual void OnBind()
        {

        }
    }
}
