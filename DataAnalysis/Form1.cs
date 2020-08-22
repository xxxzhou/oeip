using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace DataAnalysis
{
    public partial class Form1 : Form
    {
        private Setting setting = null;
        public Form1()
        {
            InitializeComponent();
            setting = SettingManager.Instance.Setting;
            //COCODataManager.Instance.LoadInstance(@"D:\WorkSpace\DeepLearning\cocodata\annotations_trainval2017\annotations\instances_train2017.json");
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            setText("");
        }

        private void button1_Click(object sender, EventArgs e)
        {
            COCODataSetting cs = new COCODataSetting();
            cs.ShowDialog();
            SettingManager.Instance.SaveSetting();
        }

        private void Form1_FormClosed(object sender, FormClosedEventArgs e)
        {
            SettingManager.Instance.SaveSetting();
        }

        public void setText(string info)
        {
            Action action = () =>
            {
                label1.Text = info;
            };
            this.Invoke(action);
        }

        public async void BuildYoloData(DataPath dataPath, string txtListName)
        {
            instances instance = new instances();
            if (!File.Exists(dataPath.AnnotationPath))
            {
                setText(dataPath.AnnotationPath + " 路径不存在.");
                return;
            }
            setText("正在读取文件中:" + Environment.NewLine + dataPath.AnnotationPath);
            var jsonTex = await Task.FromResult(File.ReadAllText(dataPath.AnnotationPath));
            setText("正在解析文件中:" + Environment.NewLine + dataPath.AnnotationPath);
            instance = await Task.FromResult(JsonConvert.DeserializeObject<instances>(jsonTex));
            setText("正在分析文件包含人物图像:" + instance.images.Count + "个");
            List<ImageLabel> labels = await Task.FromResult(COCODataManager.Instance.CreateYoloLabel(
                instance,
                (annotationOD at, image image) =>
                 {
                     //是否人类
                     return at.category_id == 1;
                 },
                (annotationOD at, image image) =>
                 {
                     //是否满足所有人类标签都面积占比都大于十分之一
                     return (at.bbox[2] / image.width) * (at.bbox[3] / image.height) > 0.1f;
                 }));
            setText("正在生成label文件:" + Environment.NewLine + dataPath.LabelPath);
            if (!Directory.Exists(dataPath.LabelPath))
            {
                Directory.CreateDirectory(dataPath.LabelPath);
            }
            await Task.Run(() =>
            {
                Parallel.ForEach(labels, (ImageLabel imageLabel) =>
                {
                    string fileName = Path.Combine(dataPath.LabelPath,
                        Path.GetFileNameWithoutExtension(imageLabel.name) + ".txt");
                    using (var file = new StreamWriter(Path.Combine(dataPath.LabelPath, fileName), false))
                    {
                        foreach (var label in imageLabel.boxs)
                        {
                            file.WriteLine(label.catId + " " + label.box.xcenter + " " + label.box.ycenter +
                                " " + label.box.width + " " + label.box.height + " ");
                        }
                    }
                });
                string path = Path.Combine(Directory.GetParent(dataPath.LabelPath).FullName, 
                    txtListName + ".txt");
                using (var file = new StreamWriter(path, false))
                {
                    foreach (var label in labels)
                    {
                        string lpath = Path.Combine(dataPath.DestImagePath, label.name);
                        file.WriteLine(lpath);
                    }
                }
            });
            setText("正在复制需要的文件到指定目录:" + dataPath.AnnotationPath);
            await Task.Run(() =>
            {
                Parallel.ForEach(labels, (ImageLabel imageLabel) =>
                {
                    string spath = Path.Combine(dataPath.SourceImagePath, imageLabel.name);
                    string dpsth = Path.Combine(dataPath.DestImagePath, imageLabel.name);
                    if (File.Exists(spath))
                        File.Copy(spath, dpsth, true);
                });
            });
            setText("全部完成");
        }

        private void button2_Click(object sender, EventArgs e)
        {
            Task.Factory.StartNew(() => BuildYoloData(setting.Data.TrainData, "train"));
        }

        private void button3_Click(object sender, EventArgs e)
        {
            Task.Factory.StartNew(() => BuildYoloData(setting.Data.PredData, "test"));
        }

        private void button4_Click(object sender, EventArgs e)
        {
            Process.Start(@"D:\WorkSpace\C++\zmf\x64\Release\LiveWindowMR.exe", " -serice");
        }
    }
}
