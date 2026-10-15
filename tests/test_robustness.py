import numpy as np
import pandas as pd
import pytest
from src.robustness import protection_forecast, queue_waits, bread_backtest, queue_benchmark
from src.module2_inventory import eoq, expected_shortage
from src.module3_newsvendor import compute_costs, BREAD_PARAMS
from src.module4_queuing import erlang_c


def test_inventory_known_values():
    assert eoq(1000,50,2)==pytest.approx(np.sqrt(50000))
    assert expected_shortage(20,np.array([10,20,40]))==pytest.approx(20/3)
    assert protection_forecast(np.full(104,10.),3,"seasonal")==pytest.approx(30)
    assert protection_forecast(np.full(104,10.),3,"empirical")==pytest.approx(30)


def test_bread_costs_and_loss_accounting():
    assert compute_costs(BREAD_PARAMS)["CR"]==pytest.approx(1.3/2.4)
    out=bread_backtest(pd.DataFrame({"SKU_E_Bread":np.full(104,100.)}))
    assert np.allclose(out.mean_loss_eur,0)


def test_queue_known_values_and_capacity_monotonicity():
    assert erlang_c(1,.5)["C_erlang"]==pytest.approx(.5)
    assert erlang_c(2,1.)["C_erlang"]==pytest.approx(1/3)
    arrivals=np.arange(30)*.2; service=np.full(30,.5)
    assert np.all(queue_waits(arrivals,service,3)<=queue_waits(arrivals,service,2))
    assert np.all(queue_waits(arrivals,np.zeros(30),1)==0)


def test_benchmark_is_reproducible_and_throws_on_bad_inputs():
    pd.testing.assert_frame_equal(queue_benchmark(1000,(2,),100),queue_benchmark(1000,(2,),100))
    with pytest.raises(ValueError): protection_forecast(np.ones(12),1,"normal")
    with pytest.raises(ValueError): queue_benchmark(10,(2,),10)
