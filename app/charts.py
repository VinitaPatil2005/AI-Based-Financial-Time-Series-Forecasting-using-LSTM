import plotly.graph_objects as go


def closing_price_chart(data):

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=data["Date"],

            y=data["Close"],

            mode="lines",

            name="Closing Price"

        )

    )

    fig.update_layout(

        title="Historical Closing Price",

        xaxis_title="Date",

        yaxis_title="Price ($)",

        template="plotly_white",

        height=500

    )

    return fig


def last30_chart(data):

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=data["Date"].tail(30),

            y=data["Close"].tail(30),

            mode="lines+markers",

            name="Last 30 Days"

        )

    )

    fig.update_layout(

        title="Last 30 Trading Days",

        template="plotly_white",

        height=450

    )

    return fig


def candlestick_chart(data):

    fig = go.Figure(

        data=[

            go.Candlestick(

                x=data["Date"],

                open=data["Open"],

                high=data["High"],

                low=data["Low"],

                close=data["Close"]

            )

        ]

    )

    fig.update_layout(

        title="Candlestick Chart",

        template="plotly_white",

        height=600

    )    

    return fig

def prediction_chart(actual, predicted):

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            y=actual.flatten(),

            name="Actual"

        )

    )

    fig.add_trace(

        go.Scatter(

            y=predicted.flatten(),

            name="Predicted"

        )

    )

    fig.update_layout(

        title="Actual vs Predicted",

        template="plotly_white",

        height=500

    )

    return fig