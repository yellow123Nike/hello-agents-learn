from dataclasses import dataclass
from typing import List, Optional


@dataclass
class File:
    """
    File 的 Docstring
    oss_url:文件在 对象存储（OSS）中的访问地址
    domain_url:文件对应的 域名访问地址，一般是对外可访问的 HTTP/HTTPS URL
    file_name:文件在系统中的 展示名称
    file_size:文件大小，单位通常为 字节
    description:文件的 文字描述或备注信息
    origin_file_name:文件的 原始文件名（用户上传时的名称
    origin_oss_url:文件在 OSS 中的 原始存储地址
    origin_domain_url:文件的 原始对外访问域名地址
    is_internal_file:标识该文件是否为 系统内部文件
    """
    oss_url: Optional[str] = None
    domain_url: Optional[str] = None
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    description: Optional[str] = None
    origin_file_name: Optional[str] = None
    origin_oss_url: Optional[str] = None
    origin_domain_url: Optional[str] = None
    is_internal_file: Optional[bool] = None


class FileUtil:

    @staticmethod
    def format_file_info(files: List[File], filter_internal_file: bool) -> str:
        """
        把一组 File 对象，按统一、可读、可直接注入 Prompt 的文本格式整理出来，并在必要时过滤掉内部文件
        """
        lines = []

        for file in files:
            # 控制“哪些文件对 Agent 可见”
            if filter_internal_file and file.is_internal_file:
                continue
            # originOssUrl 优先，其次 ossUrl
            file_url = (
                file.origin_oss_url
                if file.origin_oss_url is not None and file.origin_oss_url != ""
                else file.oss_url
            )
            lines.append(
                f"fileName:{file.file_name} "
                f"fileDesc:{file.description} "
                f"fileUrl:{file_url}"
            )

        return "\n".join(lines) + ("\n" if lines else "")


if __name__ == "__main__":
    files = [
        File(
            file_name="contract.pdf",
            description="客户合同文件",
            oss_url="oss://bucket/contract.pdf",
            origin_oss_url="https://cdn.example.com/contract.pdf",
            is_internal_file=False,
        ),
        File(
            file_name="internal.docx",
            description="系统内部说明",
            oss_url="oss://bucket/internal.docx",
            is_internal_file=True,
        ),
        File(
            file_name="report.xlsx",
            description="2024年财务报表",
            oss_url="oss://bucket/report.xlsx",
            origin_oss_url="",
            is_internal_file=False,
        ),
    ]

    # 开启内部文件过滤
    result = FileUtil.format_file_info(files, filter_internal_file=True)
    print("=== filter_internal_file=True ===")
    print(result)

    # 关闭内部文件过滤
    result_all = FileUtil.format_file_info(files, filter_internal_file=False)
    print("=== filter_internal_file=False ===")
    print(result_all)
