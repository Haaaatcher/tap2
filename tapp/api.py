import uvicorn
import click
from fastapi import FastAPI
from pydantic import BaseModel, Field
from tapp import TAPPModelInfer, TAPPInput
from typing import List
from fastapi.middleware.cors import CORSMiddleware


class TAPPOutput(BaseModel):
    value: float = Field(..., description="性能预测结果")
    unit: str | None = Field(..., description="性能单位")


app = FastAPI(title="TAPP", description="Use TAPP for property prediction of titanium alloy.", version="0.0.6")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


tapp_infer = TAPPModelInfer()


@app.get("/")
async def root():
    return {
        "app": "TAPP",
        "version": "0.0.6",
        "author": "Hang Luo",
        "email": "haaaatcher@gmail.com",
        "description": "Use TAPP for property prediction of titanium alloy."
    }


@app.post("/predict/ti_alloy/single")
async def single_predict_ta(input_data: TAPPInput) -> TAPPOutput:
    """
    单点预测钛合金性能
    """
    unit_map = {
        "TE": "10^-6/K",
        "DS": "g/cm^3",
        "TC": "W/m·K",
        "EC": "10^6 S/m",
        "YM": "GPa",
        "BM": "GPa",
        "SM": "GPa",
        "PR": None,
        "SE": "J/g",
        "SHC": "J/g·K",
        "YS": "MPa",
        "TS": "MPa",
        "HD": "VPN",
        "HP": "MPa·m^(1/2)"
    }
    output = tapp_infer(input_data)
    # 单位修正
    if input_data.Prop == "TE":
        output *= 1e6
    elif input_data.Prop == "EC":
        output *= 1e-6
    return TAPPOutput(value=output, unit=unit_map[input_data.Prop])


@app.post("/predict/ti_alloy/batch")
async def batch_predict_ta(input_data: List[TAPPInput]) -> List[TAPPOutput]:
    """
    批量预测钛合金性能（性能代号需一致）
    """
    unit_map = {
        "TE": "10^-6/K",
        "DS": "g/cm^3",
        "TC": "W/m·K",
        "EC": "10^6 S/m",
        "YM": "GPa",
        "BM": "GPa",
        "SM": "GPa",
        "PR": None,
        "SE": "J/g",
        "SHC": "J/g·K",
        "YS": "MPa",
        "TS": "MPa",
        "HD": "VPN",
        "HP": "MPa·m^(1/2)"
    }
    outputs = tapp_infer(input_data)
    results = []
    for _input, output in zip(input_data, outputs):
        # 单位修正
        if _input.Prop == "TE":
            output *= 1e6
        elif _input.Prop == "EC":
            output *= 1e-6
        results.append(TAPPOutput(value=output, unit=unit_map[_input.Prop]))
    return results


@click.command()
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8000)
def run_fast_api(host, port):
    uvicorn.run("tapp.api:app", host=host, port=port, reload=True)
