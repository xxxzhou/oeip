namespace OeipControl
{
    partial class DisplayWF
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
            this.pctBox = new System.Windows.Forms.PictureBox();
            ((System.ComponentModel.ISupportInitialize)(this.pctBox)).BeginInit();
            this.SuspendLayout();
            // 
            // pctBox
            // 
            this.pctBox.Dock = System.Windows.Forms.DockStyle.Fill;
            this.pctBox.Location = new System.Drawing.Point(0, 0);
            this.pctBox.Name = "pctBox";
            this.pctBox.Size = new System.Drawing.Size(428, 266);
            this.pctBox.TabIndex = 0;
            this.pctBox.TabStop = false;
            this.pctBox.Paint += new System.Windows.Forms.PaintEventHandler(this.pictureBox1_Paint);
            // 
            // DisplayWF
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.SystemColors.AppWorkspace;
            this.Controls.Add(this.pctBox);
            this.Name = "DisplayWF";
            this.Size = new System.Drawing.Size(428, 266);
            ((System.ComponentModel.ISupportInitialize)(this.pctBox)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.PictureBox pctBox;
    }
}
