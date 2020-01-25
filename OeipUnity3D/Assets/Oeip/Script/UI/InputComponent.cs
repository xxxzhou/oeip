
using OeipCommon.OeipAttribute;
using UnityEngine.UI;

public class InputComponent : BaseComponent<string>
{
    public InputField input;
    private InputAttribute attribute;
    // Use this for initialization
    void Start()
    {
        input.onValueChanged.AddListener(OnValueChange);
        text.text = attribute.DisplayName;
    }

    public override void OnSetAttribute()
    {
        attribute = Attribute<InputAttribute>();
        OnSetValue(attribute.Default);
    }

    public override void OnSetValue(string value)
    {
        if (attribute == null)
            return;
        this.input.text = value;
    }

    //public void OnTextChange(string text)
    //{
    //    OnValueChange(text);
    //}
}
