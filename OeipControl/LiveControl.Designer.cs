namespace OeipControl
{
    partial class LiveControl
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
            this.displayDx111 = new OeipControl.DisplayDx11();
            this.panel1.SuspendLayout();
            this.SuspendLayout();
            // 
            // panel1
            // 
            this.panel1.Controls.Add(this.displayDx111);
            this.panel1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel1.Location = new System.Drawing.Point(0, 0);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(410, 276);
            this.panel1.TabIndex = 0;
            // 
            // displayDx111
            // 
            this.displayDx111.BackColor = System.Drawing.SystemColors.AppWorkspace;
            this.displayDx111.Dock = System.Windows.Forms.DockStyle.Fill;
            this.displayDx111.Location = new System.Drawing.Point(0, 0);
            this.displayDx111.Name = "displayDx111";
            this.displayDx111.Size = new System.Drawing.Size(410, 276);
            this.displayDx111.TabIndex = 0;
            // 
            // LiveControl
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.panel1);
            this.Name = "LiveControl";
            this.Size = new System.Drawing.Size(410, 276);
            this.panel1.ResumeLayout(false);
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Panel panel1;
        private DisplayDx11 displayDx111;
    }
}
