import uvicorn
import click
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tap2.core import TAInfer, TAInput, TAOutput, TABatchInput, TABatchOutput, TAPropAbbr

_app = FastAPI(
    title="TAP2",
    description="Use TAP2 for property prediction of titanium alloy.",
    version="0.1.0"
)

_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_ta_infer_ = TAInfer()


@_app.get("/")
async def root():
    return {
        "app": "TAP2",
        "version": "0.1.0",
        "author": "Hang Luo",
        "email": "haaaatcher@gmail.com",
        "description": "Use TAP2 for property prediction of titanium alloy."
    }


@_app.post("/predict/ti_alloy/single")
async def single_predict_ta(ta_input: TAInput) -> TAOutput:
    """
    Predict titanium properties (Single)
    """
    ta_output = _ta_infer_(ta_input)
    # Unit correction
    if ta_input.prop is TAPropAbbr.TE:
        ta_output.value *= 1e6
        ta_output.unit = "10^-6/K"
    elif ta_input.prop is TAPropAbbr.EC:
        ta_output.value *= 1e-6
        ta_output.unit = "10^6 S/m"
    return ta_output


@_app.post("/predict/ti_alloy/batch")
async def batch_predict_ta(ta_input: TABatchInput) -> TABatchOutput:
    """
    Predict titanium properties (Batch)
    """
    ta_output = _ta_infer_(ta_input)
    # 单位修正
    if ta_input.prop is TAPropAbbr.TE:
        for idx in range(len(ta_output)):
            ta_output.value[idx] *= 1e6
        ta_output.unit = "10^-6/K"
    elif ta_input.prop is TAPropAbbr.EC:
        for idx in range(len(ta_output)):
            ta_output.value[idx] *= 1e-6
        ta_output.unit = "10^6 S/m"
    return ta_output


@click.command()
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8000)
def run_fast_api(host, port):
    uvicorn.run("tap2.api:_app", host=host, port=port, reload=False)
