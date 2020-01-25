using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class UITemplate : MonoBehaviour
{
    public SliderInputComponent SliderInputTemplate;
    public InputComponent InputComponent;
    public DropdownComponent DropdownComponent;
    public ToggleComponent ToggleComponent;
    // Start is called before the first frame update
    void Awake()
    {
        OeipManagerU3D.Instance.SliderInputTemplate = SliderInputTemplate;
        OeipManagerU3D.Instance.InputComponent = InputComponent;
        OeipManagerU3D.Instance.ToggleComponent = ToggleComponent;
        OeipManagerU3D.Instance.DropdownComponent = DropdownComponent;
    }
}
