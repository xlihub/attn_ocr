from pydantic import BaseModel


class InputItem(BaseModel):
    RequestId: str
    InvoiceType: str = 'dedicated_invoice'
    ImageUrl: str = None
    ImageBase64: str = None


class PaddleItem(BaseModel):
    ImageBase64: str = None
    ImageList: list = None


class SegInputItem(BaseModel):
    RequestId: str
    ImageUrl: str = None
    ImageBase64: str = None
