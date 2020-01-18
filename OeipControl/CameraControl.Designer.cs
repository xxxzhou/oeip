namespace OeipControl
{
    partial class CameraControl
    {
        /// <summary> 
        /// 必需的设计器变量。
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary> 
        /// 清理所有正在使用的资源。
        /// </summary>
        /// <param name="disposing">如果应释放托管资源，为 true；否则为 false。</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region 组件设计器生成的代码

        /// <summary> 
        /// 设计器支持所需的方法 - 不要修改
        /// 使用代码编辑器修改此方法的内容。
        /// </summary>
        private void InitializeComponent()
        {
            this.panel1 = new System.Windows.Forms.Panel();
            this.cbx_formatList = new System.Windows.Forms.ComboBox();
            this.label2 = new System.Windows.Forms.Label();
            this.cbx_cameraList = new System.Windows.Forms.ComboBox();
            this.label1 = new System.Windows.Forms.Label();
            this.panel2 = new System.Windows.Forms.Panel();
            this.label3 = new System.Windows.Forms.Label();
            this.panel3 = new System.Windows.Forms.Panel();
            this.displayDx11 = new OeipControl.DisplayDx11();
            this.displayWF = new OeipControl.DisplayWF();
            this.btn_Grabcut = new System.Windows.Forms.Button();
            this.panel1.SuspendLayout();
            this.panel2.SuspendLayout();
            this.panel3.SuspendLayout();
            this.SuspendLayout();
            // 
            // panel1
            // 
            this.panel1.BackColor = System.Drawing.SystemColors.ButtonShadow;
            this.panel1.Controls.Add(this.cbx_formatList);
            this.panel1.Controls.Add(this.label2);
            this.panel1.Controls.Add(this.cbx_cameraList);
            this.panel1.Controls.Add(this.label1);
            this.panel1.Dock = System.Windows.Forms.DockStyle.Top;
            this.panel1.Location = new System.Drawing.Point(0, 0);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(477, 33);
            this.panel1.TabIndex = 0;
            // 
            // cbx_formatList
            // 
            this.cbx_formatList.FormattingEnabled = true;
            this.cbx_formatList.Location = new System.Drawing.Point(280, 6);
            this.cbx_formatList.Name = "cbx_formatList";
            this.cbx_formatList.Size = new System.Drawing.Size(166, 20);
            this.cbx_formatList.TabIndex = 11;
            this.cbx_formatList.SelectedIndexChanged += new System.EventHandler(this.cbx_formatList_SelectedIndexChanged);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(223, 9);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(53, 12);
            this.label2.TabIndex = 10;
            this.label2.Text = "输出格式";
            // 
            // cbx_cameraList
            // 
            this.cbx_cameraList.FormattingEnabled = true;
            this.cbx_cameraList.Location = new System.Drawing.Point(51, 6);
            this.cbx_cameraList.Name = "cbx_cameraList";
            this.cbx_cameraList.Size = new System.Drawing.Size(166, 20);
            this.cbx_cameraList.TabIndex = 9;
            this.cbx_cameraList.SelectedIndexChanged += new System.EventHandler(this.cbx_cameraList_SelectedIndexChanged);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(4, 9);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(41, 12);
            this.label1.TabIndex = 8;
            this.label1.Text = "摄像机";
            // 
            // panel2
            // 
            this.panel2.BackColor = System.Drawing.SystemColors.ButtonShadow;
            this.panel2.Controls.Add(this.btn_Grabcut);
            this.panel2.Controls.Add(this.label3);
            this.panel2.Dock = System.Windows.Forms.DockStyle.Bottom;
            this.panel2.Location = new System.Drawing.Point(0, 280);
            this.panel2.Name = "panel2";
            this.panel2.Size = new System.Drawing.Size(477, 30);
            this.panel2.TabIndex = 1;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(21, 8);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(41, 12);
            this.label3.TabIndex = 0;
            this.label3.Text = "label3";
            // 
            // panel3
            // 
            this.panel3.Controls.Add(this.displayDx11);
            this.panel3.Controls.Add(this.displayWF);
            this.panel3.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel3.Location = new System.Drawing.Point(0, 33);
            this.panel3.Name = "panel3";
            this.panel3.Size = new System.Drawing.Size(477, 247);
            this.panel3.TabIndex = 2;
            // 
            // displayDx11
            // 
            this.displayDx11.BackColor = System.Drawing.SystemColors.ButtonFace;
            this.displayDx11.Dock = System.Windows.Forms.DockStyle.Bottom;
            this.displayDx11.Location = new System.Drawing.Point(0, 47);
            this.displayDx11.Name = "displayDx11";
            this.displayDx11.Size = new System.Drawing.Size(477, 200);
            this.displayDx11.TabIndex = 1;
            // 
            // displayWF
            // 
            this.displayWF.BackColor = System.Drawing.SystemColors.AppWorkspace;
            this.displayWF.Location = new System.Drawing.Point(242, 21);
            this.displayWF.Name = "displayWF";
            this.displayWF.Size = new System.Drawing.Size(217, 220);
            this.displayWF.TabIndex = 0;
            // 
            // btn_Grabcut
            // 
            this.btn_Grabcut.Location = new System.Drawing.Point(334, 3);
            this.btn_Grabcut.Name = "btn_Grabcut";
            this.btn_Grabcut.Size = new System.Drawing.Size(75, 23);
            this.btn_Grabcut.TabIndex = 1;
            this.btn_Grabcut.Text = "Grabcut";
            this.btn_Grabcut.UseVisualStyleBackColor = true;
            this.btn_Grabcut.Click += new System.EventHandler(this.btn_Grabcut_Click);
            // 
            // CameraControl
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.panel3);
            this.Controls.Add(this.panel2);
            this.Controls.Add(this.panel1);
            this.Name = "CameraControl";
            this.Size = new System.Drawing.Size(477, 310);
            this.panel1.ResumeLayout(false);
            this.panel1.PerformLayout();
            this.panel2.ResumeLayout(false);
            this.panel2.PerformLayout();
            this.panel3.ResumeLayout(false);
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.Panel panel2;
        private System.Windows.Forms.Panel panel3;
        private DisplayWF displayWF;
        private System.Windows.Forms.ComboBox cbx_formatList;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.ComboBox cbx_cameraList;
        private System.Windows.Forms.Label label1;
        private DisplayDx11 displayDx11;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Button btn_Grabcut;
    }
}
