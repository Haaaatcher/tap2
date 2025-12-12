import uvicorn
import click
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tap2.core import TAPPInfer, TAPPInput, TAPPOutput, TAPPBatchInput, TAPPBatchOutput

_app = FastAPI(title="TAP2", description="Use TAP2 for property prediction of titanium alloy.", version="0.0.7")


_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_tapp_infer = TAPPInfer()


@_app.get("/")
async def root():
    return {
        "app": "TAPP",
        "version": "0.0.7",
        "author": "Hang Luo",
        "email": "haaaatcher@gmail.com",
        "description": "Use TAPP for property prediction of titanium alloy."
    }


@_app.post("/predict/ti_alloy/single")
async def single_predict_ta(tapp_input: TAPPInput) -> TAPPOutput:
    """
    单点预测钛合金性能
    """
    tapp_output = _tapp_infer(tapp_input)
    # 单位修正
    if tapp_input.Prop == "TE":
        tapp_output.value *= 1e6
        tapp_output.unit = "10^-6/K"
    elif tapp_input.Prop == "EC":
        tapp_output.value *= 1e-6
        tapp_output.unit = "10^6 S/m"
    return tapp_output


@_app.post("/predict/ti_alloy/batch")
async def batch_predict_ta(tapp_input: TAPPBatchInput) -> TAPPBatchOutput:
    """
    批量预测钛合金性能
    """
    tapp_output = _tapp_infer(tapp_input)
    # 单位修正
    if tapp_input.Prop == "TE":
        for idx in range(len(tapp_output)):
            tapp_output.value[idx] *= 1e6
        tapp_output.unit = "10^-6/K"
    elif tapp_input.Prop == "EC":
        for idx in range(len(tapp_output)):
            tapp_output.value[idx] *= 1e-6
        tapp_output.unit = "10^6 S/m"
    return tapp_output


@click.command()
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8000)
def run_fast_api(host, port):
    uvicorn.run("tap2.api:_app", host=host, port=port, reload=False)
