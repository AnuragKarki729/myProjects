"use client";
import React, { useState } from 'react';
import { DatePicker, Space } from 'antd';
import dayjs from 'dayjs'; // Ensure dayjs is installed

const DateTimePicker = ({onChange}) => {
  const [selectedDate, setSelectedDate] = useState(null);

  const handleChange = (value, dateString) => {
    console.log('Selected Time: ', value);
    console.log('Formatted Selected Time: ', dateString);
    setSelectedDate(value);
  };

  const onOk = (value) => {
    console.log('Confirmed Time: ', value);
  };

  // Disable specific hours outside 6 AM to 10 PM range
  const disabledHours = () => {
    const hours = [];
    for (let i = 0; i < 24; i++) {
      if (i < 6 || i > 21) {
        hours.push(i); // Disable hours before 6 AM and after 9 PM
      }
    }
    return hours;
  };

  // Disable minutes other than 0 (to allow only hourly bookings)
  const disabledMinutes = () => {
    return Array.from({ length: 60 }, (_, i) => i).filter(min => min !== 0);
  };

  return (
    <Space direction="vertical" size={12}>
      <DatePicker
        showTime={{
          format: 'HH:mm',
          hideDisabledOptions: true,
          hourStep: 1,
          minuteStep: 60, // Allow only hour selection
          disabledHours,   // Disable hours outside the allowed range
          disabledMinutes, // Disable minutes other than 0
        }}
        format="YYYY-MM-DD HH:mm"
        onChange={handleChange}
        onOk={onOk}
        disabledDate={(current) => {
          // Disable past dates
          return current && current < dayjs().startOf('day');
        }}
        placeholder="Select Date and Time"
      />
    </Space>
  );
};

export default DateTimePicker;
